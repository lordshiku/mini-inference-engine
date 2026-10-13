"""Milestone 3 acceptance test (a).

For each of 8 prompts, greedy output produced under continuous batching must
equal the greedy output produced for that prompt alone. Uses a small max_batch
and varying token budgets so the scheduler genuinely admits requests mid-flight,
runs ragged-length sequences together, and evicts/compacts finished ones.
"""

import os

import pytest
import torch

from engine.generate import generate_greedy
from engine.model import Qwen2Model, load_tokenizer
from engine.scheduler import Engine

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "weights", "qwen2.5-0.5b")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Once upon a time, in a land far away,",
    "The three laws of thermodynamics are",
    "Translate to French: good morning",
    "import numpy as np\n",
    "The meaning of life is",
    "Q: What is the tallest mountain?\nA:",
]
MAX_TOKENS = [32, 40, 24, 48, 16, 36, 28, 44]


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


def test_batched_equals_standalone(model, tokenizer, device):
    # Standalone greedy for each prompt.
    standalone = []
    for prompt, mt in zip(PROMPTS, MAX_TOKENS):
        ids = torch.tensor([tokenizer.encode(prompt).ids], device=device)
        standalone.append(generate_greedy(model, ids, mt, use_cache=True, stop_on_eos=True))

    # Same prompts through the continuous-batching engine, max_batch < len(prompts)
    # so some requests wait and are admitted after others finish.
    engine = Engine(model, tokenizer=tokenizer, max_batch=4, max_len=512)
    futures = [engine.submit(prompt, mt) for prompt, mt in zip(PROMPTS, MAX_TOKENS)]
    engine.run()
    batched = [f.result() for f in futures]

    for i, (prompt, mt) in enumerate(zip(PROMPTS, MAX_TOKENS)):
        assert len(batched[i]) <= mt
        assert batched[i] == standalone[i], f"mismatch for prompt {prompt!r}"
