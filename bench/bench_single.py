"""Milestone 2 benchmark (b): single-sequence prefill and decode timing.

Reports prefill time and decode throughput (tokens/sec) with the KV cache versus
without it at a 256-token context.

The acceptance metric (cached decode >= 5x uncached) is measured as a *fair*
per-token comparison: both the cached step (project only the new token, attend
against the cached history) and the uncached step (recompute the whole 256-token
context) are CUDA-graph captured and timed as pure replays. Graph-capturing both
sides removes CPU kernel-launch overhead equally, so the remaining difference is
purely the algorithmic work the KV cache saves. For this 0.5B model an eager
(non-graphed) decode step is launch-bound, so eager throughput is reported too
for context but is not the acceptance number.
"""

import argparse
import os
import time

import torch

from engine.cache import KVCache
from engine.graph_decode import GraphDecoder
from engine.model import Qwen2Model

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")


def _sync():
    """Block until all queued CUDA work is done, so timings are accurate."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_prefill(model, ids, iters=5):
    """Return average seconds to prefill `ids` (prompt processing) with a cache."""
    for _ in range(2):  # warmup
        model.forward(ids, cache=KVCache(model.cfg.num_hidden_layers), start_pos=0)
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        model.forward(ids, cache=KVCache(model.cfg.num_hidden_layers), start_pos=0)
    _sync()
    return (time.perf_counter() - start) / iters


def time_decode_cached_eager(model, ids, decode_tokens):
    """Prefill once, then decode `decode_tokens` steps with an eager KV cache; tokens/sec."""
    cache = KVCache(model.cfg.num_hidden_layers)
    logits = model.forward(ids, cache=cache, start_pos=0)
    next_id = torch.argmax(logits[:, -1, :], dim=-1).view(1, 1)
    _sync()
    start = time.perf_counter()
    for _ in range(decode_tokens):
        logits = model.forward(next_id, cache=cache, start_pos=cache.seq_len)
        next_id = torch.argmax(logits[:, -1, :], dim=-1).view(1, 1)
    _sync()
    return decode_tokens / (time.perf_counter() - start)


def time_decode_cached_graph(decoder, ids, decode_tokens):
    """End-to-end tokens/sec generating `decode_tokens` with the CUDA-graph decoder."""
    decoder.generate(ids, decode_tokens, stop_on_eos=False)  # warmup
    _sync()
    start = time.perf_counter()
    decoder.generate(ids, decode_tokens, stop_on_eos=False)
    _sync()
    return decode_tokens / (time.perf_counter() - start)


def time_uncached_step_graph(model, ids, iters=200):
    """Tokens/sec for an uncached decode step at `len(ids)` context, CUDA-graph captured.

    Without a cache, producing one token means recomputing the full-context
    forward. We capture that full forward as a graph and time pure replays, so it
    is measured the same way as the cached step.
    """
    static_in = ids.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            model.forward(static_in, cache=None, start_pos=0)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        model.forward(static_in, cache=None, start_pos=0)
    for _ in range(5):
        graph.replay()
    _sync()
    start = time.perf_counter()
    for _ in range(iters):
        graph.replay()
    _sync()
    return iters / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", type=int, default=256)
    parser.add_argument("--decode-tokens", type=int, default=64)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = Qwen2Model.from_pretrained(MODEL_DIR, device=device, dtype=torch.float32)
    ids = torch.randint(0, model.cfg.vocab_size, (1, args.context), device=device)

    with torch.no_grad():
        prefill_s = time_prefill(model, ids)
        eager_tps = time_decode_cached_eager(model, ids, args.decode_tokens)

    decoder = GraphDecoder(model, max_len=args.context + args.decode_tokens + 8)
    graph_e2e_tps = time_decode_cached_graph(decoder, ids, args.decode_tokens)

    # Fair per-token comparison: both sides CUDA-graph captured, pure replay.
    cached_step_tps = decoder.time_decode_step(args.context)
    uncached_step_tps = time_uncached_step_graph(model, ids)
    speedup = cached_step_tps / uncached_step_tps

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}")
    print(f"context={args.context} tokens, decode={args.decode_tokens} tokens\n")
    print("| metric | value |")
    print("| --- | --- |")
    print(f"| prefill ({args.context} tok) | {prefill_s * 1000:.1f} ms |")
    print(f"| decode cached, end-to-end (CUDA graph) | {graph_e2e_tps:.1f} tok/s |")
    print(f"| decode cached, eager KV cache | {eager_tps:.1f} tok/s |")
    print(f"| per-token compute, cached step (graphed) | {cached_step_tps:.1f} tok/s |")
    print(f"| per-token compute, uncached step (graphed) | {uncached_step_tps:.1f} tok/s |")
    print(f"| cached / uncached speedup (fair, both graphed) | {speedup:.2f}x |")
    print(f"\nPASS: cached >= 5x uncached: {speedup >= 5.0}")


if __name__ == "__main__":
    main()
