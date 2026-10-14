"""Qwen2.5 forward pass in plain PyTorch.

This module loads the Qwen2.5-0.5B weights from safetensors and runs the
transformer forward pass without using the `transformers` library at runtime.
It is deliberately simple: readable building blocks (RMSNorm, RoPE, grouped
query attention, SwiGLU MLP) wired together to produce logits.
"""

import json
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer


@dataclass
class ModelConfig:
    """Hyperparameters for the Qwen2 model, read straight from config.json."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    eos_token_id: int
    max_position_embeddings: int

    @classmethod
    def from_json(cls, path):
        """Build a ModelConfig from a Qwen2 config.json file."""
        with open(path) as f:
            c = json.load(f)
        n_heads = c["num_attention_heads"]
        return cls(
            vocab_size=c["vocab_size"],
            hidden_size=c["hidden_size"],
            intermediate_size=c["intermediate_size"],
            num_hidden_layers=c["num_hidden_layers"],
            num_attention_heads=n_heads,
            num_key_value_heads=c["num_key_value_heads"],
            head_dim=c["hidden_size"] // n_heads,
            rms_norm_eps=c["rms_norm_eps"],
            rope_theta=c["rope_theta"],
            tie_word_embeddings=c["tie_word_embeddings"],
            eos_token_id=c["eos_token_id"],
            max_position_embeddings=c["max_position_embeddings"],
        )


def load_tokenizer(model_dir):
    """Load the fast tokenizer from tokenizer.json using the tokenizers library."""
    return Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))


# Milestone 4: use the fused Triton RMSNorm by default; flip to False to force
# the pure-PyTorch path (used as the correctness reference and CPU fallback).
USE_TRITON_RMSNORM = True


def rms_norm_torch(x, weight, eps):
    """Root-mean-square layer norm in pure PyTorch (the correctness reference).

    Scales each vector to unit RMS then weights it, computed in fp32 for
    numerical stability, matching the reference behaviour.
    """
    dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(var + eps)
    return weight * x.to(dtype)


def rms_norm(x, weight, eps):
    """RMSNorm dispatcher: fused Triton kernel when enabled and on CUDA, else PyTorch.

    This is what the forward passes call, so enabling the kernel transparently
    replaces the PyTorch implementation everywhere (M4 requirement).
    """
    if USE_TRITON_RMSNORM and x.is_cuda:
        from engine.kernels import rms_norm_triton

        return rms_norm_triton(x, weight, eps)
    return rms_norm_torch(x, weight, eps)


def build_rope_cache(head_dim, max_pos, theta, device, dtype):
    """Precompute RoPE cos/sin tables of shape (max_pos, head_dim).

    Uses the GPT-NeoX / HF layout where the frequencies are concatenated with
    themselves so that rotate_half pairs element i with element i+head_dim/2.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x):
    """Rotate the two halves of the last dim: (x1, x2) -> (-x2, x1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    """Apply rotary position embeddings to query and key tensors.

    q, k have shape (B, n_heads, T, head_dim); cos/sin have shape (T, head_dim).
    """
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def repeat_kv(x, n_rep):
    """Repeat KV heads n_rep times so grouped-query heads align with query heads.

    x: (B, n_kv_heads, T, head_dim) -> (B, n_kv_heads * n_rep, T, head_dim).
    """
    if n_rep == 1:
        return x
    b, n_kv, t, d = x.shape
    x = x[:, :, None, :, :].expand(b, n_kv, n_rep, t, d)
    return x.reshape(b, n_kv * n_rep, t, d)


class Qwen2Model:
    """The Qwen2.5 transformer: embeddings -> decoder layers -> final norm -> LM head.

    Weights live in plain dicts/tensors rather than nn.Module parameters to keep
    the forward pass explicit and easy to follow.
    """

    def __init__(self, config, weights, device):
        """Store config/weights and precompute the RoPE tables in the model dtype."""
        self.cfg = config
        self.device = device
        self.w = weights
        self.dtype = weights["model.embed_tokens.weight"].dtype
        max_pos = config.max_position_embeddings
        # RoPE tables in the model's dtype so decode stays in that dtype end to end
        # (fp32 for the logit test, bf16 for fast inference).
        self.cos, self.sin = build_rope_cache(
            config.head_dim, max_pos, config.rope_theta, device, self.dtype
        )

    @classmethod
    def from_pretrained(cls, model_dir, device="cuda", dtype=torch.float32):
        """Load config and safetensors weights from a model directory."""
        cfg = ModelConfig.from_json(os.path.join(model_dir, "config.json"))
        raw = load_file(os.path.join(model_dir, "model.safetensors"))
        weights = {k: v.to(device=device, dtype=dtype) for k, v in raw.items()}
        return cls(cfg, weights, device)

    def _attention(self, x, layer, cos, sin, cache):
        """Grouped-query attention with a causal mask for one decoder layer.

        When `cache` is given, the new tokens' keys/values are appended to it and
        attention runs against the full (past + new) key/value history, so decode
        steps only project the new token rather than the whole sequence.
        """
        cfg = self.cfg
        b, t, _ = x.shape
        p = f"model.layers.{layer}.self_attn."
        q = F.linear(x, self.w[p + "q_proj.weight"], self.w[p + "q_proj.bias"])
        k = F.linear(x, self.w[p + "k_proj.weight"], self.w[p + "k_proj.bias"])
        v = F.linear(x, self.w[p + "v_proj.weight"], self.w[p + "v_proj.bias"])

        q = q.view(b, t, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = k.view(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = v.view(b, t, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        if cache is not None:
            k, v = cache.update(layer, k, v)

        n_rep = cfg.num_attention_heads // cfg.num_key_value_heads
        k = repeat_kv(k, n_rep)
        v = repeat_kv(v, n_rep)

        s = k.shape[2]
        scale = cfg.head_dim ** -0.5
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        scores = scores + self._causal_mask(t, s, scores.dtype, x.device)
        probs = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        return F.linear(out, self.w[p + "o_proj.weight"])

    @staticmethod
    def _causal_mask(t, s, dtype, device):
        """Additive mask (t, s): the t new queries are the last t of s keys.

        Query i (the (s - t + i)-th position) may attend to key j only if
        j <= s - t + i. For prefill (s == t) this is the usual triangular mask;
        for a single-token decode step (t == 1) nothing is masked.
        """
        q_idx = torch.arange(t, device=device).view(t, 1) + (s - t)
        k_idx = torch.arange(s, device=device).view(1, s)
        allowed = k_idx <= q_idx
        mask = torch.zeros((t, s), dtype=dtype, device=device)
        mask.masked_fill_(~allowed, float("-inf"))
        return mask

    def _mlp(self, x, layer):
        """SwiGLU feed-forward: down(silu(gate(x)) * up(x))."""
        p = f"model.layers.{layer}.mlp."
        gate = F.linear(x, self.w[p + "gate_proj.weight"])
        up = F.linear(x, self.w[p + "up_proj.weight"])
        return F.linear(F.silu(gate) * up, self.w[p + "down_proj.weight"])

    def forward(self, input_ids, cache=None, start_pos=0):
        """Run the forward pass and return logits of shape (B, T, vocab).

        `start_pos` is the absolute position of the first input token (0 for a
        fresh prompt, or the current sequence length when decoding with a cache),
        so RoPE and the causal mask use correct positions.
        """
        cfg = self.cfg
        x = F.embedding(input_ids, self.w["model.embed_tokens.weight"])
        t = input_ids.shape[1]
        cos = self.cos[start_pos:start_pos + t]
        sin = self.sin[start_pos:start_pos + t]

        for layer in range(cfg.num_hidden_layers):
            ln1 = self.w[f"model.layers.{layer}.input_layernorm.weight"]
            residual = x
            x = rms_norm(x, ln1, cfg.rms_norm_eps)
            x = residual + self._attention(x, layer, cos, sin, cache)

            ln2 = self.w[f"model.layers.{layer}.post_attention_layernorm.weight"]
            residual = x
            x = rms_norm(x, ln2, cfg.rms_norm_eps)
            x = residual + self._mlp(x, layer)

        x = rms_norm(x, self.w["model.norm.weight"], cfg.rms_norm_eps)

        if cfg.tie_word_embeddings:
            lm_head = self.w["model.embed_tokens.weight"]
        else:
            lm_head = self.w["lm_head.weight"]
        return F.linear(x, lm_head)
