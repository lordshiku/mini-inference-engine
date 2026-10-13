"""Milestone 2 acceptance test (a).

Greedy generation of 64 tokens with the KV cache must equal generation without
the cache, token-for-token, on 5 prompts.
"""

import os

import pytest
import torch

from engine.generate import generate_greedy
from engine.graph_decode import GraphDecoder
from engine.model import Qwen2Model, load_tokenizer

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "The three laws of thermodynamics are",
    "Translate to French: good morning",
]


@pytest.fixture(scope="module")
def device():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return "cuda"


@pytest.fixture(scope="module")
def model(device):
    return Qwen2Model.from_pretrained(MODEL_DIR, device=device, dtype=torch.float32)


@pytest.fixture(scope="module")
def tokenizer():
    return load_tokenizer(MODEL_DIR)


def test_cache_matches_no_cache(model, tokenizer, device):
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        with torch.no_grad():
            cached = generate_greedy(model, ids, 64, use_cache=True, stop_on_eos=False)
            uncached = generate_greedy(model, ids, 64, use_cache=False, stop_on_eos=False)
        assert len(cached) == 64 and len(uncached) == 64
        assert cached == uncached, f"mismatch for prompt {prompt!r}"


def test_graph_decode_matches_eager(model, tokenizer, device):
    # The CUDA-graph fast decoder must produce the same greedy tokens as eager.
    decoder = GraphDecoder(model, max_len=256)
    for prompt in PROMPTS:
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        with torch.no_grad():
            eager = generate_greedy(model, ids, 64, use_cache=True, stop_on_eos=False)
            graphed = decoder.generate(ids, 64, stop_on_eos=False)
        assert graphed == eager, f"graph decode mismatch for prompt {prompt!r}"
