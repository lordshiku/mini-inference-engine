"""Triton kernels.

A fused RMSNorm: one Triton program per row (one token's hidden vector) loads the
row once, computes the mean square and reciprocal std, and writes the normalized,
weighted output — fusing what would otherwise be several separate PyTorch kernels
(square, mean, rsqrt, multiply) into a single launch. Normalization is done in
fp32 for stability, matching the PyTorch reference.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_fwd(x_ptr, w_ptr, out_ptr, row_stride, n_cols, eps,
                  BLOCK: tl.constexpr):
    """One program normalizes one row of length n_cols (padded to BLOCK)."""
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * row_stride + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * rstd * w
    tl.store(out_ptr + row * row_stride + cols, y, mask=mask)


def rms_norm_triton(x, weight, eps):
    """Fused RMSNorm over the last dim of `x` using the Triton kernel.

    `x` has shape (..., H) and `weight` has shape (H,); returns a tensor of the
    same shape/dtype as `x`. Falls back nowhere — callers choose PyTorch vs Triton.
    """
    h = x.shape[-1]
    x2d = x.contiguous().view(-1, h)
    out = torch.empty_like(x2d)
    block = triton.next_power_of_2(h)
    _rms_norm_fwd[(x2d.shape[0],)](
        x2d, weight.contiguous(), out, x2d.stride(0), h, eps, BLOCK=block,
    )
    return out.view(x.shape)
