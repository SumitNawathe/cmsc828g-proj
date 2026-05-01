"""
gpu_generate.py — GPU generation strategy implementations.

Each strategy is a callable with the signature:

    fn(model, tokenizer, samples, G, B) -> dict

and is registered in GPU_GENERATE_REGISTRY under a short name.
Pass the name via --gen-strategy to run_inference.py.

Current strategies
------------------
recursive_retry  :  recursive_retry_gpu_generate
    Tiles each prompt G times, runs a single model.generate() call,
    and handles OOM by splitting the batch in half and retrying.
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any


def _build_prompt(tokenizer: AutoTokenizer, question: str) -> str:
    """Build a chat-formatted prompt for a single question."""
    messages = [
        {"role": "user", "content": f"Please think step by step, and provide your answer in \\boxed{{}}. Question: {question}"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def recursive_retry_gpu_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    G: int,
    B: int,
) -> dict[str, Any]:
    """
    Generate G rollouts for each sample in the batch (basic strategy).

    Args:
        model: The loaded language model (already on the correct GPU).
        tokenizer: The tokenizer for the model.
        samples: List of dicts, each with at least a 'question' key.
        G: Number of rollouts per sample.
        B: Maximum number of new tokens to generate.

    Returns:
        dict with keys:
          - decoded_outputs:      list[list[str]]   shape [num_samples, G]
          - output_token_lengths: list[list[int]]    shape [num_samples, G]
          - generation_time:      float (wall-clock seconds)
          - oom_occurred:         bool
    """
    start_all = time.perf_counter()

    if not samples:
        return {
            "decoded_outputs": [],
            "output_token_lengths": [],
            "generation_time": time.perf_counter() - start_all,
            "oom_occurred": False,
        }

    # Build prompts and tokenize
    prompts = [_build_prompt(tokenizer, s["question"]) for s in samples]

    # Tile each prompt G times so we get G rollouts per sample in one generate call
    tiled_prompts = []
    for p in prompts:
        tiled_prompts.extend([p] * G)

    # Tokenize with padding (left-pad for generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        tiled_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    oom = False
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=B,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        oom = True

    if oom:
        # We handle OOM *outside* the except block. 
        # If we handle it inside the except block, Python's sys.exc_info() 
        # keeps the traceback alive, which holds references to all local 
        # variables in the HuggingFace generate() stack (like huge KV caches),
        # preventing empty_cache() from freeing them!
        del inputs
        torch.cuda.empty_cache()

        if len(samples) <= 1:
            # Cannot split further — a single sample OOMs.
            # Generate with G=1 (one rollout at a time) as last resort.
            if G > 1:
                all_decoded: list[list[str]] = [[] for _ in samples]
                all_lengths: list[list[int]] = [[] for _ in samples]
                for g_idx in range(G):
                    result = recursive_retry_gpu_generate(model, tokenizer, samples, G=1, B=B)
                    for s_idx in range(len(samples)):
                        all_decoded[s_idx].extend(result["decoded_outputs"][s_idx])
                        all_lengths[s_idx].extend(result["output_token_lengths"][s_idx])
                return {
                    "decoded_outputs": all_decoded,
                    "output_token_lengths": all_lengths,
                    "generation_time": time.perf_counter() - start_all,
                    "oom_occurred": True,
                }
            else:
                # Even a single rollout OOMs — re-raise
                raise torch.cuda.OutOfMemoryError("OOM even with G=1 for a single sample")

        # Split batch in half and retry
        mid = len(samples) // 2
        left_samples = samples[:mid]
        right_samples = samples[mid:]

        left_result = recursive_retry_gpu_generate(model, tokenizer, left_samples, G, B)
        right_result = recursive_retry_gpu_generate(model, tokenizer, right_samples, G, B)

        return {
            "decoded_outputs": left_result["decoded_outputs"] + right_result["decoded_outputs"],
            "output_token_lengths": left_result["output_token_lengths"] + right_result["output_token_lengths"],
            "generation_time": time.perf_counter() - start_all,
            "oom_occurred": True,
        }

    # Only decode the NEW tokens (skip input tokens)
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    decoded_outputs: list[list[str]] = []
    output_token_lengths: list[list[int]] = []

    for i, sample in enumerate(samples):
        sample_decoded = []
        sample_lengths = []
        for g in range(G):
            flat_idx = i * G + g
            inp_len = int(input_lengths[flat_idx])
            new_tokens = outputs[flat_idx][inp_len:]
            num_new = len(new_tokens)
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            sample_decoded.append(text)
            sample_lengths.append(num_new)
        decoded_outputs.append(sample_decoded)
        output_token_lengths.append(sample_lengths)

    # Free GPU memory eagerly
    del outputs, inputs
    torch.cuda.empty_cache()

    return {
        "decoded_outputs": decoded_outputs,
        "output_token_lengths": output_token_lengths,
        "generation_time": time.perf_counter() - start_all,
        "oom_occurred": False,
    }


# ─── Registry ────────────────────────────────────────────────────────────────
# Maps strategy name (used in --gen-strategy CLI arg) to the callable.
# Add new strategies here as they are implemented.

GPU_GENERATE_REGISTRY: dict[str, Any] = {
    "recursive_retry": recursive_retry_gpu_generate,
}
