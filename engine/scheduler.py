"""Continuous batching scheduler.

An `Engine` holds a queue of requests and a batched KV cache. Its run loop, each
iteration: admits waiting requests into free batch rows (prefilling them), runs
one batched greedy decode step for all active sequences (which may be at ragged
lengths), then evicts finished sequences and compacts the batch. This keeps the
GPU busy across many concurrent sequences, which is where throughput comes from.
"""

from collections import deque
from concurrent.futures import Future

import torch
import torch.nn.functional as F

from engine.cache import BatchedKVCache, KVCache
from engine.model import repeat_kv, rms_norm, rotate_half


class Request:
    """One generation request: its prompt, token budget, output, and a Future."""

    def __init__(self, prompt_ids, max_tokens, stop_on_eos):
        """Record the prompt token ids and how many tokens to generate."""
        self.prompt_ids = prompt_ids
        self.max_tokens = max_tokens
        self.stop_on_eos = stop_on_eos
        self.generated = []
        self.next_token = None  # the token to feed on the next decode step
        self.row = None         # batch row this request occupies while active
        self.future = Future()


class Engine:
    """Continuous-batching inference engine for a single model."""

    def __init__(self, model, tokenizer=None, max_batch=16, max_len=1024):
        """Allocate the batched KV cache and set up the request queue."""
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_len = max_len
        cfg = model.cfg
        self.cache = BatchedKVCache(
            cfg.num_hidden_layers, max_batch, cfg.num_key_value_heads, max_len,
            cfg.head_dim, model.device, model.w["model.embed_tokens.weight"].dtype,
        )
        self.queue = deque()
        self.active = []

    def submit(self, prompt, max_tokens, stop_on_eos=True):
        """Queue a request and return a Future that run() resolves to the token ids.

        `prompt` may be a string (tokenized with the engine's tokenizer) or a list
        of token ids. The Future's result is the list of generated token ids.
        """
        if isinstance(prompt, str):
            if self.tokenizer is None:
                raise ValueError("string prompt requires a tokenizer")
            prompt_ids = self.tokenizer.encode(prompt).ids
        else:
            prompt_ids = list(prompt)
        req = Request(prompt_ids, max_tokens, stop_on_eos)
        self.queue.append(req)
        return req.future

    def _prefill(self, req, row):
        """Prefill one request into `row`; return its first generated token id.

        Uses the single-sequence forward (proven correct in M1/M2) and copies the
        resulting K/V into the batched cache row.
        """
        ids = torch.tensor([req.prompt_ids], device=self.model.device)
        cache = KVCache(self.model.cfg.num_hidden_layers)
        logits = self.model.forward(ids, cache=cache, start_pos=0)
        self.cache.write_prefill(row, cache.k, cache.v, len(req.prompt_ids))
        return torch.argmax(logits[:, -1, :], dim=-1).item()

    def _decode_step(self, n_active):
        """Run one batched greedy decode step over active rows [0, n_active).

        Feeds each sequence its pending token, writes that token's K/V at the
        sequence's current position, attends over the (ragged) history with a
        per-row mask, and returns the predicted next tokens (n_active,).
        """
        cfg, w, m = self.model.cfg, self.model.w, self.model
        rows = torch.arange(n_active, device=m.device)
        positions = self.cache.lengths[:n_active]  # position of each fed token
        tokens = torch.tensor([[r.next_token] for r in self.active],
                              device=m.device)  # (B, 1)

        x = F.embedding(tokens, w["model.embed_tokens.weight"])
        # Per-row RoPE tables, already shaped to broadcast over heads and the
        # single query position (so we inline the rotation rather than using
        # apply_rope, which assumes a shared (T, head_dim) table).
        cos = m.cos[positions].view(n_active, 1, 1, cfg.head_dim)
        sin = m.sin[positions].view(n_active, 1, 1, cfg.head_dim)

        s = int(positions.max().item()) + 1  # attend only over filled columns
        col = torch.arange(s, device=m.device).view(1, s)
        valid = col <= positions.view(n_active, 1)  # (B, s)
        mask = torch.zeros((n_active, 1, 1, s), device=m.device)
        mask.masked_fill_(~valid.view(n_active, 1, 1, s), float("-inf"))

        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        scale = cfg.head_dim ** -0.5
        for l in range(cfg.num_hidden_layers):
            p = f"model.layers.{l}.self_attn."
            residual = x
            xn = rms_norm(x, w[f"model.layers.{l}.input_layernorm.weight"], cfg.rms_norm_eps)
            q = F.linear(xn, w[p + "q_proj.weight"], w[p + "q_proj.bias"])
            k = F.linear(xn, w[p + "k_proj.weight"], w[p + "k_proj.bias"])
            v = F.linear(xn, w[p + "v_proj.weight"], w[p + "v_proj.bias"])
            q = q.view(n_active, 1, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
            k = k.view(n_active, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            v = v.view(n_active, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            self.cache.write_token(l, rows, positions, k[:, :, 0, :], v[:, :, 0, :])
            kk = repeat_kv(self.cache.k[l, :n_active, :, :s, :], n_rep)
            vv = repeat_kv(self.cache.v[l, :n_active, :, :s, :], n_rep)
            scores = torch.matmul(q, kk.transpose(-1, -2)) * scale + mask
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(n_active, 1, -1)
            x = residual + F.linear(o, w[p + "o_proj.weight"])

            residual = x
            xn = rms_norm(x, w[f"model.layers.{l}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            pp = f"model.layers.{l}.mlp."
            gate = F.linear(xn, w[pp + "gate_proj.weight"])
            up = F.linear(xn, w[pp + "up_proj.weight"])
            x = residual + F.linear(F.silu(gate) * up, w[pp + "down_proj.weight"])

        x = rms_norm(x, w["model.norm.weight"], cfg.rms_norm_eps)
        lm_head = w["model.embed_tokens.weight"] if cfg.tie_word_embeddings else w["lm_head.weight"]
        logits = F.linear(x, lm_head)
        self.cache.lengths[:n_active] += 1
        return torch.argmax(logits[:, -1, :], dim=-1)

    def run(self):
        """Process all queued requests to completion, resolving each Future.

        Each iteration admits as many waiting requests as there is room for,
        decodes one token for every active sequence, then finishes/evicts any
        sequence that hit its token budget or produced EOS.
        """
        eos = self.model.cfg.eos_token_id
        with torch.no_grad():
            while self.queue or self.active:
                while self.queue and len(self.active) < self.max_batch:
                    req = self.queue.popleft()
                    row = len(self.active)
                    req.row = row
                    req.next_token = self._prefill(req, row)
                    self.active.append(req)

                preds = self._decode_step(len(self.active))

                survivors = []
                for i, req in enumerate(self.active):
                    req.generated.append(req.next_token)
                    finished = (len(req.generated) >= req.max_tokens
                                or (req.stop_on_eos and req.next_token == eos))
                    if finished:
                        req.future.set_result(req.generated)
                    else:
                        req.next_token = preds[i].item()
                        survivors.append(req)

                survivor_rows = [req.row for req in survivors]
                self.cache.compact(survivor_rows)
                for new_row, req in enumerate(survivors):
                    req.row = new_row
                self.active = survivors
