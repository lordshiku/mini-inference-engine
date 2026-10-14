"""Milestone 3/6 benchmark: continuous-batching throughput vs concurrency.

Submits N identical requests to the Engine, runs them to completion, and reports
total throughput (tokens/sec across all sequences) at concurrency 1, 4, 8, 16,
32. Each concurrency level uses its own Engine with max_batch = concurrency and
the CUDA-graph decode path, so the captured graph is sized to that level (a
concurrency-1 run does not pay for 32 rows). M6 runs this in bf16 by default.

Acceptance (M3) requires throughput at 16 >= 3x throughput at 1.
"""

import argparse
import os
import time

import torch

from engine.model import Qwen2Model, load_tokenizer
from engine.scheduler import Engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")
PROMPT = "Once upon a time, in a land far away,"

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def throughput_at(model, tokenizer, concurrency, max_tokens, use_cuda_graph):
    """Run `concurrency` identical requests through a fresh Engine; return tokens/sec.

    The engine's max_batch is set to the concurrency so the CUDA graph is sized to
    this level. One warmup run() captures the graph and is not timed.
    """
    prompt_ids = tokenizer.encode(PROMPT).ids
    engine = Engine(model, tokenizer=tokenizer, max_batch=concurrency,
                    max_len=len(prompt_ids) + max_tokens + 8, use_cuda_graph=use_cuda_graph)

    # Warmup (captures the graph); not timed.
    wf = [engine.submit(list(prompt_ids), max_tokens, stop_on_eos=False) for _ in range(concurrency)]
    engine.run()
    for f in wf:
        f.result()

    futures = [engine.submit(list(prompt_ids), max_tokens, stop_on_eos=False)
               for _ in range(concurrency)]
    torch.cuda.synchronize()
    start = time.perf_counter()
    engine.run()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(f.result()) for f in futures)
    return total_tokens / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    parser.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    parser.add_argument("--no-cuda-graph", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2Model.from_pretrained(MODEL_DIR, device=device, dtype=DTYPES[args.dtype])
    tokenizer = load_tokenizer(MODEL_DIR)
    use_graph = (not args.no_cuda_graph) and torch.cuda.is_available()

    results = {}
    for c in args.concurrency:
        results[c] = throughput_at(model, tokenizer, c, args.max_tokens, use_graph)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}")
    print(f"dtype={args.dtype}, cuda_graph={use_graph}, "
          f"prompt tokens={len(tokenizer.encode(PROMPT).ids)}, max_tokens={args.max_tokens}\n")
    print("| concurrency | throughput (tok/s) | speedup vs 1 |")
    print("| --- | --- | --- |")
    base = results[args.concurrency[0]]
    for c in args.concurrency:
        print(f"| {c} | {results[c]:.1f} | {results[c] / base:.2f}x |")

    if 16 in results and 1 in results:
        ratio = results[16] / results[1]
        print(f"\nPASS: throughput@16 >= 3x throughput@1: {ratio >= 3.0} ({ratio:.2f}x)")


if __name__ == "__main__":
    main()
