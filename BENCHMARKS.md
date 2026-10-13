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
