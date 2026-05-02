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

pruned_kernel    :  pruned_kernel_generate
    Same tiling strategy, but replaces model.generate() with a
    custom token-by-token loop (gpu_generate_efficient._run_pruned_loop)
    that drops finished sequences from the active batch mid-generation.
    Uses a Triton sampling kernel when available; falls back to PyTorch.
    On OOM, splits batch in half and retries recursively.

pruned_kernel_safe  :  pruned_kernel_generate_safe
    Like pruned_kernel, but proactively monitors GPU memory every K tokens
    and offloads half the batch to CPU when usage exceeds a threshold.
    Avoids OOM without discarding partial work.  Extra hyperparameters:
    K (chunk size, default 32) and mem_threshold (default 0.95).
"""

import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any

from gpu_generate_efficient import _run_pruned_loop, _run_pruned_loop_safe


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
        print(f"  [OOM] recursive_retry: n_samples={len(samples)} G={G} — clearing cache",
              flush=True)
        del inputs
        torch.cuda.empty_cache()

        if len(samples) <= 1:
            # Cannot split samples further — halve G instead.
            if G > 1:
                half_g = G // 2
                remainder_g = G - half_g
                print(f"  [OOM] Single sample, halving G: {G} → {half_g} + {remainder_g}",
                      flush=True)
                left_result = recursive_retry_gpu_generate(
                    model, tokenizer, samples, G=half_g, B=B,
                )
                right_result = recursive_retry_gpu_generate(
                    model, tokenizer, samples, G=remainder_g, B=B,
                )
                return {
                    "decoded_outputs": [
                        l + r for l, r in zip(
                            left_result["decoded_outputs"],
                            right_result["decoded_outputs"],
                        )
                    ],
                    "output_token_lengths": [
                        l + r for l, r in zip(
                            left_result["output_token_lengths"],
                            right_result["output_token_lengths"],
                        )
                    ],
                    "generation_time": time.perf_counter() - start_all,
                    "oom_occurred": True,
                }
            else:
                # Even a single rollout OOMs — re-raise
                raise torch.cuda.OutOfMemoryError("OOM even with G=1 for a single sample")

        # Split batch in half and retry
        mid = len(samples) // 2
        print(f"  [OOM] Splitting batch: {len(samples)} → {mid} + {len(samples)-mid}",
              flush=True)
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

    # Only decode the NEW tokens (skip input tokens).
    # model.generate() right-pads all sequences to the same length, so we
    # decode with skip_special_tokens=True then re-encode to get the true
    # per-sequence output length.
    input_lengths = inputs["attention_mask"].sum(dim=1).tolist()

    decoded_outputs: list[list[str]] = []
    output_token_lengths: list[list[int]] = []

    for i in range(len(samples)):
        sample_decoded = []
        sample_lengths = []
        for g in range(G):
            flat_idx = i * G + g
            inp_len = int(input_lengths[flat_idx])
            new_tokens = outputs[flat_idx][inp_len:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True)
            num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            sample_decoded.append(text)
            sample_lengths.append(num_tokens)
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


def pruned_kernel_generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    G: int,
    B: int,
) -> dict[str, Any]:
    """
    Generate G rollouts for each sample using dynamic batch pruning + Triton
    sampling (efficient strategy).

    Uses a custom token-by-token loop (see gpu_generate_efficient) that removes
    sequences from the active batch the moment they emit an EOS token, reducing
    both GPU memory and compute as generation progresses.

    Args:
        model:     The loaded language model (already on the correct GPU).
        tokenizer: The tokenizer for the model.
        samples:   List of dicts, each with at least a 'question' key.
        G:         Number of rollouts per sample.
        B:         Maximum number of new tokens to generate.

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

    # Build prompts and tile G times so we get G rollouts per sample
    prompts = [_build_prompt(tokenizer, s["question"]) for s in samples]
    tiled_prompts = [p for p in prompts for _ in range(G)]

    # Tokenize with left-padding (required for batched causal-LM generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        tiled_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    eos_ids = set()
    eos_raw = tokenizer.eos_token_id
    if isinstance(eos_raw, int):
        eos_ids.add(eos_raw)
    elif isinstance(eos_raw, list):
        eos_ids.update(eos_raw)

    oom = False
    try:
        flat_generated = _run_pruned_loop(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=B,
            eos_ids=eos_ids,
        )
    except torch.cuda.OutOfMemoryError:
        oom = True

    if oom:
        # We handle OOM *outside* the except block so that Python's
        # sys.exc_info() does not keep the traceback alive — the traceback
        # holds references to all local variables in the generate() stack
        # (e.g. large KV caches), preventing empty_cache() from freeing them.
        print(f"  [OOM] pruned_kernel: n_samples={len(samples)} G={G} — clearing cache",
              flush=True)
        del inputs
        torch.cuda.empty_cache()

        if len(samples) <= 1:
            if G > 1:
                # Single sample OOMs — halve G and retry
                half_g = G // 2
                remainder_g = G - half_g
                print(f"  [OOM] Single sample, halving G: {G} → {half_g} + {remainder_g}",
                      flush=True)
                left_result = pruned_kernel_generate(
                    model, tokenizer, samples, G=half_g, B=B,
                )
                right_result = pruned_kernel_generate(
                    model, tokenizer, samples, G=remainder_g, B=B,
                )
                return {
                    "decoded_outputs": [
                        l + r for l, r in zip(
                            left_result["decoded_outputs"],
                            right_result["decoded_outputs"],
                        )
                    ],
                    "output_token_lengths": [
                        l + r for l, r in zip(
                            left_result["output_token_lengths"],
                            right_result["output_token_lengths"],
                        )
                    ],
                    "generation_time": time.perf_counter() - start_all,
                    "oom_occurred": True,
                }
            else:
                raise torch.cuda.OutOfMemoryError("OOM even with G=1 for a single sample")

        # Split batch in half and retry
        mid = len(samples) // 2
        print(f"  [OOM] Splitting batch: {len(samples)} → {mid} + {len(samples)-mid}",
              flush=True)
        left_result = pruned_kernel_generate(model, tokenizer, samples[:mid], G, B)
        right_result = pruned_kernel_generate(model, tokenizer, samples[mid:], G, B)

        return {
            "decoded_outputs": left_result["decoded_outputs"] + right_result["decoded_outputs"],
            "output_token_lengths": left_result["output_token_lengths"] + right_result["output_token_lengths"],
            "generation_time": time.perf_counter() - start_all,
            "oom_occurred": True,
        }

    # Decode and reshape from flat [num_samples * G] → [num_samples, G]
    decoded_outputs: list[list[str]] = []
    output_token_lengths: list[list[int]] = []

    for i in range(len(samples)):
        sample_decoded = []
        sample_lengths = []
        for g in range(G):
            token_ids = flat_generated[i * G + g]
            # Strip trailing EOS tokens before decoding
            while token_ids and token_ids[-1] in eos_ids:
                token_ids = token_ids[:-1]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            sample_decoded.append(text)
            sample_lengths.append(num_tokens)
        decoded_outputs.append(sample_decoded)
        output_token_lengths.append(sample_lengths)

    del inputs
    torch.cuda.empty_cache()

    return {
        "decoded_outputs": decoded_outputs,
        "output_token_lengths": output_token_lengths,
        "generation_time": time.perf_counter() - start_all,
        "oom_occurred": False,
    }


def pruned_kernel_generate_safe(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: list[dict[str, Any]],
    G: int,
    B: int,
    K: int = 32,
    mem_threshold: float = 0.95,
) -> dict[str, Any]:
    """
    Generate G rollouts per sample with proactive GPU memory management.

    Like pruned_kernel_generate, but checks GPU memory every K tokens and
    offloads half the active batch to CPU when usage exceeds mem_threshold.
    Deferred batches are reloaded and completed sequentially, avoiding OOM
    without discarding any partial work.

    Args:
        model:         The loaded language model (already on the correct GPU).
        tokenizer:     The tokenizer for the model.
        samples:       List of dicts, each with at least a 'question' key.
        G:             Number of rollouts per sample.
        B:             Maximum number of new tokens to generate.
        K:             Tokens per memory-check chunk (default 32).
        mem_threshold: GPU memory fraction (0–1) that triggers batch split
                       (default 0.95).

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

    # Build prompts and tile G times so we get G rollouts per sample
    prompts = [_build_prompt(tokenizer, s["question"]) for s in samples]
    tiled_prompts = [p for p in prompts for _ in range(G)]

    # Tokenize with left-padding (required for batched causal-LM generation)
    tokenizer.padding_side = "left"
    inputs = tokenizer(
        tiled_prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    eos_ids = set()
    eos_raw = tokenizer.eos_token_id
    if isinstance(eos_raw, int):
        eos_ids.add(eos_raw)
    elif isinstance(eos_raw, list):
        eos_ids.update(eos_raw)

    oom = False
    try:
        flat_generated = _run_pruned_loop_safe(
            model=model,
            tokenizer=tokenizer,
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=B,
            eos_ids=eos_ids,
            K=K,
            mem_threshold=mem_threshold,
        )
    except torch.cuda.OutOfMemoryError:
        oom = True

    if oom:
        # Handle OOM outside the except block so the traceback doesn't pin
        # large KV-cache tensors (same pattern as recursive_retry).
        print(f"  [OOM] pruned_kernel_safe: n_samples={len(samples)} G={G} — clearing cache",
              flush=True)
        del inputs
        torch.cuda.empty_cache()

        if len(samples) <= 1:
            if G > 1:
                half_g = G // 2
                remainder_g = G - half_g
                print(f"  [OOM] Single sample, halving G: {G} → {half_g} + {remainder_g}",
                      flush=True)
                left_result = pruned_kernel_generate_safe(
                    model, tokenizer, samples, G=half_g, B=B,
                    K=K, mem_threshold=mem_threshold,
                )
                right_result = pruned_kernel_generate_safe(
                    model, tokenizer, samples, G=remainder_g, B=B,
                    K=K, mem_threshold=mem_threshold,
                )
                return {
                    "decoded_outputs": [
                        l + r for l, r in zip(
                            left_result["decoded_outputs"],
                            right_result["decoded_outputs"],
                        )
                    ],
                    "output_token_lengths": [
                        l + r for l, r in zip(
                            left_result["output_token_lengths"],
                            right_result["output_token_lengths"],
                        )
                    ],
                    "generation_time": time.perf_counter() - start_all,
                    "oom_occurred": True,
                }
            else:
                raise torch.cuda.OutOfMemoryError("OOM even with G=1 for a single sample")

        mid = len(samples) // 2
        print(f"  [OOM] Splitting batch: {len(samples)} → {mid} + {len(samples)-mid}",
              flush=True)
        left_result = pruned_kernel_generate_safe(
            model, tokenizer, samples[:mid], G, B, K=K, mem_threshold=mem_threshold,
        )
        right_result = pruned_kernel_generate_safe(
            model, tokenizer, samples[mid:], G, B, K=K, mem_threshold=mem_threshold,
        )

        return {
            "decoded_outputs": left_result["decoded_outputs"] + right_result["decoded_outputs"],
            "output_token_lengths": left_result["output_token_lengths"] + right_result["output_token_lengths"],
            "generation_time": time.perf_counter() - start_all,
            "oom_occurred": True,
        }

    # Decode and reshape from flat [num_samples * G] → [num_samples, G]
    decoded_outputs: list[list[str]] = []
    output_token_lengths: list[list[int]] = []

    for i in range(len(samples)):
        sample_decoded = []
        sample_lengths = []
        for g in range(G):
            token_ids = flat_generated[i * G + g]
            while token_ids and token_ids[-1] in eos_ids:
                token_ids = token_ids[:-1]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)
            num_tokens = len(tokenizer.encode(text, add_special_tokens=False))
            sample_decoded.append(text)
            sample_lengths.append(num_tokens)
        decoded_outputs.append(sample_decoded)
        output_token_lengths.append(sample_lengths)

    del inputs
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
    "pruned_kernel": pruned_kernel_generate,
    "pruned_kernel_safe": pruned_kernel_generate_safe,
}
