# Benchmarks

GPU: **NVIDIA RTX A5000**. All numbers in fp32. Reproduce with the commands shown.

## Milestone 2 — KV cache (single sequence)

`PYTHONPATH=. python bench/bench_single.py` — 2026-10-13, NVIDIA RTX A5000

Context = 256 tokens, decode = 64 tokens.

| metric | value |
| --- | --- |
| prefill (256 tok) | 27.5 ms |
| decode cached, end-to-end (CUDA graph) | 189.9 tok/s |
| decode cached, eager KV cache | 46.1 tok/s |
| per-token compute, cached step (graphed) | 210.7 tok/s |
| per-token compute, uncached step (graphed) | 37.1 tok/s |
| **cached / uncached speedup (fair, both graphed)** | **5.69x** |

Acceptance (cached decode >= 5x uncached at 256-token context): **PASS**.

### How the speedup is measured (and why it is fair)

The acceptance ratio compares the per-token *compute* of a cached decode step
(project only the new token, attend against the cached K/V history) against an
uncached step (recompute the full 256-token forward, as you must with no cache).

Both sides are CUDA-graph captured and timed as pure replays. This matters
because for a 0.5B model an eager decode step is dominated by CPU kernel-launch
overhead (~430 launches/step), not compute — eager cached decode is only ~1.5x
eager uncached, which would understate the cache. Graph-capturing *both* sides
removes that launch overhead equally, so the remaining 5.69x is purely the
algorithmic work the KV cache saves (1 new token vs 256 recomputed tokens).

The CUDA-graph decoder is numerically identical to eager cached decode, verified
token-for-token in `tests/test_cache.py::test_graph_decode_matches_eager`. The
uncached step uses a fixed 256-token context, so the ratio is conservative (real
uncached generation grows the context and gets slower).

Reproducibility: two consecutive runs agreed within <2% on every number
(speedup 5.69x then 5.65x).

## Milestone 3 — continuous batching (throughput vs concurrency)

`PYTHONPATH=. python bench/bench_concurrent.py` — 2026-10-13, NVIDIA RTX A5000

Prompt = 11 tokens, max_tokens = 64, max_batch = concurrency. Throughput is total
tokens/sec across all sequences.

| concurrency | throughput (tok/s) | speedup vs 1 |
| --- | --- | --- |
| 1 | 45.5 | 1.00x |
| 4 | 171.6 | 3.77x |
| 8 | 324.6 | 7.13x |
| 16 | 585.4 | 12.86x |
| 32 | 917.3 | 20.14x |

Acceptance (throughput@16 >= 3x throughput@1): **PASS** (12.86x).

Continuous batching (not the fallback static batching): the scheduler admits
waiting requests into free rows each step and evicts/compacts finished ones, so
rows are reused mid-run. The strong scaling comes from this model being
launch-bound per step — folding more sequences into one batched step adds little
step cost, so tokens/sec rises nearly linearly with the batch until compute
starts to bind around concurrency 32.

Reproducibility: a second run gave 46.2 / 176.2 / 329.9 / 582.9 / 944.4 tok/s
(throughput@16 = 12.62x), within ~3% of the above on every level.

## Milestone 4 — Triton fused RMSNorm

`PYTHONPATH=. python bench/bench_kernels.py` — 2026-10-13, NVIDIA RTX A5000

Fused Triton RMSNorm vs the PyTorch implementation it replaces (one program per
row; square/mean/rsqrt/scale fused into a single launch). Per-call latency, timed
with CUDA events (min over 5 blocks of 500 calls):

| shape (rows x hidden) | PyTorch (us) | Triton (us) | speedup |
| --- | --- | --- | --- |
| 1 x 896 | 55.6 | 35.7 | 1.56x |
| 32 x 896 | 56.2 | 35.1 | 1.60x |
| 256 x 896 | 59.2 | 36.0 | 1.64x |
| 4096 x 896 | 166.0 | 45.2 | 3.68x |

The kernel wins at every size by collapsing several PyTorch kernel launches into
one; the gap widens with row count as the PyTorch path becomes bandwidth-bound
while the fused kernel stays efficient. Correctness is verified within atol 1e-4
over 10+ shapes in `tests/test_kernels.py`, and the kernel is enabled by default
(`engine.model.USE_TRITON_RMSNORM`), so all earlier milestone tests run through
it and still pass (6 passed).

Reproducibility: a second run gave 1.55 / 1.56 / 1.59 / 3.67x, within ~3% of the
above on every shape.
