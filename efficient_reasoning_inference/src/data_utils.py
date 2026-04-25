"""
MATH500 dataset loading and Qwen3 prompt formatting utilities.
"""
from __future__ import annotations

from typing import List, Tuple

from datasets import load_dataset


def load_math500_sample(n_samples: int = 50, seed: int = 42) -> List[dict]:
    """Load a random subset of MATH500 problems (HuggingFaceH4/MATH-500)."""
    dataset = load_dataset("HuggingFaceH4/MATH-500", split="test")
    dataset = dataset.shuffle(seed=seed)
    n = min(n_samples, len(dataset))
    return [dataset[i] for i in range(n)]


def format_prompt(problem: dict, tokenizer, enable_thinking: bool = True) -> str:
    """
    Format a MATH500 problem as a Qwen3 chat prompt.

    Tries the Qwen3-specific `enable_thinking` kwarg; falls back gracefully
    for tokenizers that do not support it.
    """
    messages = [{"role": "user", "content": problem["problem"]}]
    try:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return text


def get_prompts(
    n_samples: int,
    tokenizer,
    seed: int = 42,
    enable_thinking: bool = True,
) -> Tuple[List[str], List[dict]]:
    """Load MATH500 problems and return formatted prompts alongside raw dicts."""
    problems = load_math500_sample(n_samples, seed=seed)
    prompts = [format_prompt(p, tokenizer, enable_thinking=enable_thinking) for p in problems]
    return prompts, problems
