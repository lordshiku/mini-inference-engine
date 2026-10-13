"""Milestone 4 acceptance test (a).

The Triton fused RMSNorm must match the PyTorch RMSNorm within atol 1e-4 over 10
random shapes (and a couple of fixed ones, including the model's hidden size).
"""

import pytest
import torch

from engine.kernels import rms_norm_triton
from engine.model import rms_norm_torch


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


# 10 random (rows, hidden) shapes plus the model's hidden size (896).
SHAPES = [
    (1, 896), (3, 896), (7, 128), (16, 512), (32, 1024),
    (2, 2048), (5, 777), (64, 256), (128, 64), (4, 4096),
]


def test_rms_norm_matches_pytorch(device):
    eps = 1e-6
    torch.manual_seed(0)
    for rows, hidden in SHAPES:
        x = torch.randn(rows, hidden, device=device, dtype=torch.float32)
        weight = torch.randn(hidden, device=device, dtype=torch.float32)
        ref = rms_norm_torch(x, weight, eps)
        got = rms_norm_triton(x, weight, eps)
        assert torch.allclose(got, ref, atol=1e-4), (
            f"shape {(rows, hidden)}: max diff {(got - ref).abs().max().item()}"
        )


def test_rms_norm_3d_shape(device):
    # The forward pass calls RMSNorm on (B, T, H) tensors; make sure that works.
    eps = 1e-6
    x = torch.randn(2, 5, 896, device=device, dtype=torch.float32)
    weight = torch.randn(896, device=device, dtype=torch.float32)
    ref = rms_norm_torch(x, weight, eps)
    got = rms_norm_triton(x, weight, eps)
    assert torch.allclose(got, ref, atol=1e-4)
