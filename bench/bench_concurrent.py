"""Milestone 3 benchmark (b): continuous-batching throughput vs concurrency.

Submits N identical requests to the Engine, runs them to completion, and reports
total throughput (tokens/sec across all sequences) at concurrency 1, 4, 8, 16,
32. Because this 0.5B model is CPU-launch-bound per step, batching many
sequences into one step raises throughput roughly with the batch size. Acceptance
requires throughput at 16 >= 3x throughput at 1.
"""

import argparse
import os
import time

import torch

from engine.model import Qwen2Model, load_tokenizer
from engine.scheduler import Engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")
PROMPT = "Once upon a time, in a land far away,"


def throughput_at(engine, tokenizer, concurrency, max_tokens):
    """Run `concurrency` identical requests and return total tokens/sec."""
    prompt_ids = tokenizer.encode(PROMPT).ids
    futures = [engine.submit(list(prompt_ids), max_tokens, stop_on_eos=False)
               for _ in range(concurrency)]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    engine.run()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    total_tokens = sum(len(f.result()) for f in futures)
    return total_tokens / elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Qwen2Model.from_pretrained(MODEL_DIR, device=device, dtype=torch.float32)
    tokenizer = load_tokenizer(MODEL_DIR)
    max_batch = max(args.concurrency)
    engine = Engine(model, tokenizer=tokenizer, max_batch=max_batch,
                    max_len=len(tokenizer.encode(PROMPT).ids) + args.max_tokens + 8)

    # Warmup so the first measurement is not penalised by lazy init.
    throughput_at(engine, tokenizer, 2, 8)

    results = {}
    for c in args.concurrency:
        results[c] = throughput_at(engine, tokenizer, c, args.max_tokens)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    print(f"GPU: {gpu}")
    print(f"prompt tokens={len(tokenizer.encode(PROMPT).ids)}, max_tokens={args.max_tokens}\n")
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
