"""CUDA-graph single-sequence greedy decode.

Eager decode launches hundreds of tiny kernels per token (dozens per layer), so
for a 0.5B model the step is dominated by CPU launch overhead, not compute, and
the KV cache's O(1)-per-token advantage over O(n) recompute barely shows. This
module captures one decode step as a CUDA graph over a fixed-size ("static") KV
cache, collapsing those launches into a single replay. The result is numerically
identical to eager cached decode but fast enough to make the cache win visible.
"""

import time

import torch
import torch.nn.functional as F

from engine.cache import KVCache
from engine.model import apply_rope, repeat_kv, rms_norm


class GraphDecoder:
    """Fast greedy decoder: eager prefill, then CUDA-graph decode steps.

    The KV cache is preallocated to `max_len` so every decode step has identical
    tensor shapes (a requirement for graph capture); positions beyond the current
    token are masked out. The graph is captured once and reused for every prompt
    by overwriting the static input/cache buffers in place before each replay.
    """

    def __init__(self, model, max_len):
        """Allocate static buffers sized to `max_len` and capture the decode graph."""
        if not torch.cuda.is_available():
            raise RuntimeError("GraphDecoder requires CUDA")
        self.m = model
        cfg = model.cfg
        self.max_len = max_len
        dev = model.device
        nl, nkv, hd = cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim
        dt = model.dtype

        # Static KV cache and per-step inputs that the captured graph reads from.
        self.kbuf = torch.zeros(nl, 1, nkv, max_len, hd, device=dev, dtype=dt)
        self.vbuf = torch.zeros(nl, 1, nkv, max_len, hd, device=dev, dtype=dt)
        self.tok = torch.zeros(1, 1, dtype=torch.long, device=dev)
        self.pos_idx = torch.zeros(1, dtype=torch.long, device=dev)
        self.cos_b = torch.zeros(1, hd, device=dev, dtype=dt)
        self.sin_b = torch.zeros(1, hd, device=dev, dtype=dt)
        self.mask = torch.full((1, 1, 1, max_len), float("-inf"), device=dev, dtype=dt)
        self._capture()

    def _decode_once(self):
        """One decode step over the static buffers; returns logits (1, 1, vocab)."""
        m, cfg, w = self.m, self.m.cfg, self.m.w
        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        scale = cfg.head_dim ** -0.5
        x = F.embedding(self.tok, w["model.embed_tokens.weight"])

        for l in range(cfg.num_hidden_layers):
            p = f"model.layers.{l}.self_attn."
            residual = x
            xn = rms_norm(x, w[f"model.layers.{l}.input_layernorm.weight"], cfg.rms_norm_eps)
            q = F.linear(xn, w[p + "q_proj.weight"], w[p + "q_proj.bias"])
            k = F.linear(xn, w[p + "k_proj.weight"], w[p + "k_proj.bias"])
            v = F.linear(xn, w[p + "v_proj.weight"], w[p + "v_proj.bias"])
            q = q.view(1, 1, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
            k = k.view(1, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            v = v.view(1, 1, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
            q, k = apply_rope(q, k, self.cos_b, self.sin_b)

            # Write this token's K/V into its slot, then attend over the whole buffer.
            self.kbuf[l].index_copy_(2, self.pos_idx, k)
            self.vbuf[l].index_copy_(2, self.pos_idx, v)
            kk = repeat_kv(self.kbuf[l], n_rep)
            vv = repeat_kv(self.vbuf[l], n_rep)
            scores = torch.matmul(q, kk.transpose(-1, -2)) * scale + self.mask
            probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
            o = torch.matmul(probs, vv).transpose(1, 2).reshape(1, 1, -1)
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

    def _capture(self):
        """Warm up on a side stream, then capture one decode step as a CUDA graph."""
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                self._decode_once()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            self.out = self._decode_once()

    def _reset(self):
        """Clear the static cache/mask so a fresh prompt starts from an empty history."""
        self.kbuf.zero_()
        self.vbuf.zero_()
        self.mask.fill_(float("-inf"))

    def generate(self, input_ids, max_new_tokens, stop_on_eos=True):
        """Greedy-generate up to `max_new_tokens` tokens for one prompt.

        Prefills the prompt eagerly into the static cache, then runs graph-replay
        decode steps. Returns the list of generated token ids (excluding prompt).
        """
        cfg = self.m.cfg
        prompt_len = input_ids.shape[1]
        if prompt_len + max_new_tokens > self.max_len:
            raise ValueError("prompt + max_new_tokens exceeds GraphDecoder max_len")

        self._reset()
        with torch.no_grad():
            prefill = KVCache(cfg.num_hidden_layers)
            logits = self.m.forward(input_ids, cache=prefill, start_pos=0)
            for l in range(cfg.num_hidden_layers):
                self.kbuf[l][:, :, :prompt_len] = prefill.k[l]
                self.vbuf[l][:, :, :prompt_len] = prefill.v[l]
            next_id = torch.argmax(logits[:, -1, :], dim=-1)

            generated = []
            for i in range(max_new_tokens):
                pos = prompt_len + i
                token = next_id.item()
                generated.append(token)
                if stop_on_eos and token == cfg.eos_token_id:
                    break
                if i == max_new_tokens - 1:
                    break
                self.tok.copy_(next_id.view(1, 1))
                self.pos_idx.fill_(pos)
                self.cos_b.copy_(self.m.cos[pos].view(1, -1))
                self.sin_b.copy_(self.m.sin[pos].view(1, -1))
                self.mask[..., :pos + 1] = 0.0
                self.graph.replay()
                next_id = torch.argmax(self.out[:, -1, :], dim=-1)

        return generated

    def time_decode_step(self, context_len, iters=200):
        """Benchmark helper: tokens/sec for one decode step at `context_len` context.

        Primes the static cache to `context_len` filled positions and replays the
        captured decode graph `iters` times with no host-side sampling, isolating
        the per-token compute the KV cache enables (one new token vs recomputing
        the whole context). Returns tokens/sec.
        """
        self._reset()
        self.tok.zero_()
        self.pos_idx.fill_(context_len)
        self.cos_b.copy_(self.m.cos[context_len].view(1, -1))
        self.sin_b.copy_(self.m.sin[context_len].view(1, -1))
        self.mask[..., :context_len + 1] = 0.0
        for _ in range(5):
            self.graph.replay()
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            self.graph.replay()
        torch.cuda.synchronize()
        return iters / (time.perf_counter() - start)
