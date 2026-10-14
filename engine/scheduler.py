"""Continuous batching scheduler.

An `Engine` holds a queue of requests and a batched KV cache. Its run loop, each
iteration: admits waiting requests into free batch rows (prefilling them), runs
one batched greedy decode step for all active sequences (which may be at ragged
lengths), then evicts finished sequences and compacts the batch. This keeps the
GPU busy across many concurrent sequences, which is where throughput comes from.

The batched decode step can run eagerly or (default, on CUDA) through a captured
CUDA graph. Graph replay collapses the step's hundreds of kernel launches into
one, which matters a lot for this launch-bound 0.5B model. The graph needs static
shapes, so it always runs `max_batch` rows (inactive rows are harmless dummies)
and attends over the full `max_len` cache with a per-row mask.
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

    def __init__(self, model, tokenizer=None, max_batch=16, max_len=1024,
                 use_cuda_graph=True):
        """Allocate the batched KV cache and set up the request queue."""
        self.model = model
        self.tokenizer = tokenizer
        self.max_batch = max_batch
        self.max_len = max_len
        self.dtype = model.dtype
        self.use_cuda_graph = use_cuda_graph and torch.cuda.is_available()
        cfg = model.cfg
        self.cache = BatchedKVCache(
            cfg.num_hidden_layers, max_batch, cfg.num_key_value_heads, max_len,
            cfg.head_dim, model.device, self.dtype,
        )
        self.queue = deque()
        self.active = []
        self._graphs = {}  # batch size -> (CUDAGraph, logits output tensor)
        if self.use_cuda_graph:
            self._alloc_graph_buffers()

    # ------------------------------------------------------------------ API
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

    # -------------------------------------------------------------- internals
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

    def _run_layers(self, tokens, cos, sin, mask, rows, positions, n, s):
        """Shared batched decode compute over `n` rows, attending to `s` key columns.

        Feeds one token per row, writes its K/V into the cache at each row's
        position, and returns logits of shape (n, 1, vocab). Used by both the
        eager path and the CUDA-graph capture (which passes the static buffers).
        """
        cfg, w = self.model.cfg, self.model.w
        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        scale = cfg.head_dim ** -0.5
        x = F.embedding(tokens, w["model.embed_tokens.weight"])

        for l in range(cfg.num_hidden_layers):
            p = f"model.layers.{l}.self_attn."
            residual = x
            xn = rms_norm(x, w[f"model.layers.{l}.input_layernorm.weight"], cfg.rms_norm_eps)
            q = F.linear(xn, w[p + "q_proj.weight"], w[p + "q_proj.bias"])
            k = F.linear(xn, w[p + "k_proj.weight"], w[p + "k_proj.bias"])
            v = F.linear(xn, w[p + "v_proj.weight"], w[p + "v_proj.bias"])
            q = q.view(n, 1, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
            k = k.view(n, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            v = v.view(n, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            q = q * cos + rotate_half(q) * sin
            k = k * cos + rotate_half(k) * sin

            self.cache.write_token(l, rows, positions, k[:, :, 0, :], v[:, :, 0, :])
            kk = repeat_kv(self.cache.k[l, :n, :, :s, :], n_rep)
            vv = repeat_kv(self.cache.v[l, :n, :, :s, :], n_rep)
            scores = torch.matmul(q, kk.transpose(-1, -2)) * scale + mask
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(n, 1, -1)
            x = residual + F.linear(o, w[p + "o_proj.weight"])

            residual = x
            xn = rms_norm(x, w[f"model.layers.{l}.post_attention_layernorm.weight"], cfg.rms_norm_eps)
            pp = f"model.layers.{l}.mlp."
            gate = F.linear(xn, w[pp + "gate_proj.weight"])
            up = F.linear(xn, w[pp + "up_proj.weight"])
            x = residual + F.linear(F.silu(gate) * up, w[pp + "down_proj.weight"])

        x = rms_norm(x, w["model.norm.weight"], cfg.rms_norm_eps)
        lm_head = w["model.embed_tokens.weight"] if cfg.tie_word_embeddings else w["lm_head.weight"]
        return F.linear(x, lm_head)

    def _decode_step(self, n_active):
        """Run one batched greedy decode step; return predicted next tokens (n_active,)."""
        if self.use_cuda_graph:
            return self._decode_step_graph(n_active)
        return self._decode_step_eager(n_active)

    def _decode_step_eager(self, n_active):
        """Eager batched decode over active rows [0, n_active) (fallback / CPU path)."""
        m, cfg = self.model, self.model.cfg
        hd = cfg.head_dim
        rows = torch.arange(n_active, device=m.device)
        positions = self.cache.lengths[:n_active]
        tokens = torch.tensor([[r.next_token] for r in self.active], device=m.device)
        cos = m.cos[positions].view(n_active, 1, 1, hd)
        sin = m.sin[positions].view(n_active, 1, 1, hd)

        s = int(positions.max().item()) + 1
        col = torch.arange(s, device=m.device).view(1, s)
        valid = col <= positions.view(n_active, 1)
        mask = torch.zeros((n_active, 1, 1, s), device=m.device, dtype=self.dtype)
        mask.masked_fill_(~valid.view(n_active, 1, 1, s), float("-inf"))

        logits = self._run_layers(tokens, cos, sin, mask, rows, positions, n_active, s)
        self.cache.lengths[:n_active] += 1
        return torch.argmax(logits[:, -1, :], dim=-1)

    # ------------------------------------------------------------ CUDA graph
    def _alloc_graph_buffers(self):
        """Allocate the static per-step input buffers the captured graph reads from."""
        b = self.max_batch
        hd = self.model.cfg.head_dim
        dev, dt = self.model.device, self.dtype
        self.g_tokens = torch.zeros(b, 1, dtype=torch.long, device=dev)
        self.g_positions = torch.zeros(b, dtype=torch.long, device=dev)
        self.g_cos = torch.zeros(b, 1, 1, hd, device=dev, dtype=dt)
        self.g_sin = torch.zeros(b, 1, 1, hd, device=dev, dtype=dt)
        self.g_mask = torch.full((b, 1, 1, self.max_len), float("-inf"), device=dev, dtype=dt)
        self.g_rows = torch.arange(b, device=dev)

    def _decode_compute(self):
        """Run the full-batch decode over the static buffers (the graphed region)."""
        return self._run_layers(self.g_tokens, self.g_cos, self.g_sin, self.g_mask,
                                self.g_rows, self.g_positions, self.max_batch, self.max_len)

    def _capture_graph(self, b):
        """Warm up on a side stream, then capture a decode step graph for batch `b`."""
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                self._decode_compute()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            logits = self._decode_compute()
        self._graphs[b] = (graph, logits)

    def _decode_step_graph(self, n_active):
        """Batched decode via CUDA graph: fill static buffers, replay, read logits.

        The graph always runs `max_batch` rows; rows >= n_active are dummies whose
        outputs are ignored (and whose scattered K/V land in inactive rows that are
        overwritten on the next admission's prefill).
        """
        b = self.max_batch
        self.g_positions.zero_()
        self.g_positions[:n_active] = self.cache.lengths[:n_active]
        toks = [r.next_token for r in self.active] + [0] * (b - n_active)
        self.g_tokens.copy_(torch.tensor(toks, device=self.model.device).view(b, 1))
        hd = self.model.cfg.head_dim
        self.g_cos.copy_(self.model.cos[self.g_positions].view(b, 1, 1, hd))
        self.g_sin.copy_(self.model.sin[self.g_positions].view(b, 1, 1, hd))
        col = torch.arange(self.max_len, device=self.model.device).view(1, self.max_len)
        valid = col <= self.g_positions.view(b, 1)
        self.g_mask.fill_(float("-inf"))
        self.g_mask.masked_fill_(valid.view(b, 1, 1, self.max_len), 0.0)

        if b not in self._graphs:
            self._capture_graph(b)
        # Always replay to read results: the compute executed during capture does
        # not leave reliable values in the output tensor. Replaying re-scatters the
        # same K/V at the same slots (idempotent), so the cache stays correct.
        self._graphs[b][0].replay()
        logits = self._graphs[b][1]

        self.cache.lengths[:n_active] += 1
        return torch.argmax(logits[:n_active, -1, :], dim=-1)
