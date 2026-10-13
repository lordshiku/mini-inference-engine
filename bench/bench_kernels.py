"""Milestone 4 benchmark (b): Triton fused RMSNorm vs PyTorch eager.

Times both implementations on representative shapes (the model hidden size 896
over a range of row counts, i.e. prefill-like and decode-like workloads) and
reports the kernel speedup. Prints a markdown table.
"""

import os

import torch

from engine.kernels import rms_norm_triton
from engine.model import rms_norm_torch

SHAPES = [
    (1, 896),      # single decode token
    (32, 896),     # batched decode (concurrency 32)
    (256, 896),    # prefill, 256-token context
    (4096, 896),   # large batch * sequence
]
EPS = 1e-6


def _time(fn, iters=500, repeats=5):
    """Return the min per-call seconds over `repeats` timed blocks of `iters`.

    Uses CUDA events (precise GPU-side timing) and takes the minimum across
    repeats, which filters scheduler/clock noise and makes microsecond-scale
    latencies reproducible run-to-run.
    """
    for _ in range(20):  # warmup
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        start_evt.record()
        for _ in range(iters):
            fn()
        end_evt.record()
        torch.cuda.synchronize()
        best = min(best, start_evt.elapsed_time(end_evt) / 1e3 / iters)
    return best


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"

    print(f"GPU: {gpu}\n")
    print("| shape (rows x hidden) | PyTorch (us) | Triton (us) | speedup |")
    print("| --- | --- | --- | --- |")
    for rows, hidden in SHAPES:
        x = torch.randn(rows, hidden, device=device, dtype=torch.float32)
        w = torch.randn(hidden, device=device, dtype=torch.float32)
        t_torch = _time(lambda: rms_norm_torch(x, w, EPS))
        t_triton = _time(lambda: rms_norm_triton(x, w, EPS))
        print(f"| {rows} x {hidden} | {t_torch * 1e6:.1f} | {t_triton * 1e6:.1f} "
              f"| {t_torch / t_triton:.2f}x |")


if __name__ == "__main__":
    main()
