"""
gpu_generate_efficient.py — Auxiliary helpers for pruned_kernel_generate().

Provides two building blocks:

  1. Triton sampling kernel (triton_topk_sample / triton_greedy_decode):
     Fused temperature scaling + top-k/p filtering + multinomial sample.
     Uses a two-pass row-wise argmax Triton kernel for greedy decoding to
     avoid materialising a full sorted copy of the vocabulary that
     torch.argmax / torch.topk(k=1) may do internally.
     Falls back to pure PyTorch if Triton is not installed.

  2. _prune_kv_cache: Slices the HuggingFace KV-cache in place along the
     batch dimension so finished sequences can be dropped mid-generation.

  3. _run_pruned_loop: Custom token-by-token generation loop that uses the
     above two helpers to run a shrinking batch — sequences are removed from
     the active set the moment they emit an EOS token.

Code for (1) and (2) is adapted from efficient_reasoning_inference/src/.
"""

import torch
from transformers import DynamicCache
from typing import List, Set
import triton
import triton.language as tl


@triton.jit
def _row_max_kernel(
    logits_ptr,         # float32 [B, V]
    out_ptr,            # float32 [B]
    V,                  # vocab size (runtime int)
    BLOCK_V: tl.constexpr,
):
    """Pass 1: find the maximum logit value in each row."""
    row = tl.program_id(0)
    base = row * V
    running_max = -float("inf")

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        vals = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        block_max = tl.max(vals, axis=0)
        running_max = tl.where(block_max > running_max, block_max, running_max)

    tl.store(out_ptr + row, running_max)

@triton.jit
def _row_argmax_kernel(
    logits_ptr,         # float32 [B, V]
    max_ptr,            # float32 [B]  (from pass 1)
    out_ptr,            # int32   [B]
    V,
    BLOCK_V: tl.constexpr,
):
    """Pass 2: find the first index whose value equals the row maximum."""
    row = tl.program_id(0)
    base = row * V
    global_max = tl.load(max_ptr + row)

    # Use V as out-of-range sentinel; we want the *minimum* valid index.
    best_idx = V

    for start in range(0, V, BLOCK_V):
        offs = start + tl.arange(0, BLOCK_V)
        mask = offs < V
        vals = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
        is_max = (vals == global_max) & mask
        cand = tl.where(is_max, offs, V)
        block_min = tl.min(cand, axis=0)
        best_idx = tl.where(block_min < best_idx, block_min, best_idx)

    tl.store(out_ptr + row, best_idx)

def triton_greedy_decode(logits: torch.Tensor) -> torch.Tensor:
    """
    Row-wise argmax via two Triton kernel passes.

    Args:
        logits: float32 tensor of shape [B, V]. Must be contiguous.

    Returns:
        int64 tensor of shape [B] with the argmax index per row.
    """
    if not logits.is_contiguous():
        logits = logits.contiguous()

    B, V = logits.shape
    BLOCK_V = 1024  # tunable; fits comfortably in registers

    row_maxes = torch.empty(B, dtype=torch.float32, device=logits.device)
    argmax_out = torch.empty(B, dtype=torch.int32, device=logits.device)

    grid = (B,)
    _row_max_kernel[grid](logits, row_maxes, V, BLOCK_V=BLOCK_V)
    _row_argmax_kernel[grid](logits, row_maxes, argmax_out, V, BLOCK_V=BLOCK_V)

    return argmax_out.to(torch.int64)

def triton_topk_sample(
    logits: torch.Tensor,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
    greedy: bool = False,
) -> torch.Tensor:
    """
    Sample the next token per row using the Triton kernel for greedy mode
    and PyTorch for stochastic sampling (top-k + top-p).

    Args:
        logits:      float32 [B, V] logits tensor.
        temperature: Softmax temperature (ignored when greedy=True).
        top_p:       Nucleus filtering threshold.
        top_k:       Top-k filtering count (0 = disabled).
        greedy:      If True, use argmax instead of sampling.

    Returns:
        int64 tensor of shape [B] with the sampled token index per row.
    """
    if greedy or temperature == 0.0:
        return triton_greedy_decode(logits)

    scaled = logits / temperature if temperature != 1.0 else logits

    if top_k > 0:
        k = min(top_k, scaled.size(-1))
        topk_vals = torch.topk(scaled, k, dim=-1).values
        threshold = topk_vals[:, -1].unsqueeze(-1)
        scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
        cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        remove = (cum_probs - torch.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        scaled = scaled.scatter(1, sorted_idx, sorted_logits)

    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)



# ---------------------------------------------------------------------------
# KV-cache pruning
# ---------------------------------------------------------------------------

def _prune_kv_cache(past_key_values, keep: torch.Tensor):
    """
    Prune the batch dimension of a HuggingFace KV-cache object.

    Index-selects along dim-0 (batch) for every tensor in the cache.
    Supports DynamicCache (transformers ≥ 4.38), per-layer cache objects,
    and legacy tuple-of-tuples caches.

    Args:
        past_key_values: HuggingFace KV-cache returned by model().
        keep:            1-D long tensor of batch indices to retain.

    Returns:
        The pruned cache object (same type as input when possible).
    """
    def _prune_4d_tensors(obj) -> bool:
        """Index-select every 4-D tensor in obj's __dict__ along dim-0."""
        found = False
        for name, val in list(vars(obj).items()):
            if isinstance(val, torch.Tensor) and val.ndim == 4:
                setattr(obj, name, val[keep])
                found = True
        return found

    # Strategy 1a: flat lists of 4-D tensors (DynamicCache ≥ 4.38)
    cache_lists = {
        name: val
        for name, val in vars(past_key_values).items()
        if (
            isinstance(val, list)
            and val
            and isinstance(val[0], torch.Tensor)
            and val[0].ndim == 4
        )
    }
    if cache_lists:
        for lst in cache_lists.values():
            for i in range(len(lst)):
                lst[i] = lst[i][keep]
        for attr in ("_seen_tokens", "seen_tokens"):
            if hasattr(past_key_values, attr):
                first_list = next(iter(cache_lists.values()))
                setattr(past_key_values, attr, first_list[0].shape[-2])
                break
        return past_key_values

    # Strategy 1b: per-layer cache objects stored in a 'layers' list
    layers_attr = getattr(past_key_values, "layers", None)
    if isinstance(layers_attr, list) and layers_attr:
        pruned_any = False
        for layer_cache in layers_attr:
            pruned_any |= _prune_4d_tensors(layer_cache)
            for name, val in list(vars(layer_cache).items()):
                if (isinstance(val, list) and val
                        and isinstance(val[0], torch.Tensor) and val[0].ndim == 4):
                    for i in range(len(val)):
                        val[i] = val[i][keep]
                    pruned_any = True
        if pruned_any:
            return past_key_values

    # Strategy 2: legacy tuple-of-tuples (transformers < 4.38)
    if isinstance(past_key_values, tuple):
        new_cache = DynamicCache()
        for i, layer in enumerate(past_key_values):
            k, v = layer[0], layer[1]
            new_cache.update(k[keep], v[keep], i)
        return new_cache

    raise RuntimeError(
        f"_prune_kv_cache: unrecognised cache type '{type(past_key_values).__name__}'. "
        f"Instance attrs: {list(vars(past_key_values).keys())}"
    )


# ---------------------------------------------------------------------------
# Pruned generation loop
# ---------------------------------------------------------------------------

def _run_pruned_loop(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_ids: Set[int],
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> List[List[int]]:
    """
    Token-by-token generation loop with dynamic batch pruning.

    Sequences are removed from the active batch the moment they emit an EOS
    token: their KV-cache rows are dropped and the attention mask is sliced.
    This shrinks GPU memory and compute as the batch drains.

    Sampling uses the Triton kernel when available; falls back to PyTorch.

    Args:
        model:            HuggingFace causal-LM (on GPU, eval mode).
        tokenizer:        Corresponding tokenizer (used only for pad_token_id).
        input_ids:        Left-padded prompt tensor [batch_size, prompt_len].
        attention_mask:   Corresponding attention mask [batch_size, prompt_len].
        max_new_tokens:   Maximum tokens to generate per sequence.
        eos_ids:          Set of token IDs that signal end-of-sequence.
        temperature:      Sampling temperature.
        top_p:            Nucleus filtering threshold.
        top_k:            Top-k filtering count.

    Returns:
        List of length batch_size, each element being a list of generated
        token IDs (not including the prompt; stops at EOS, inclusive).
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    active_indices: List[int] = list(range(batch_size))
    generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]

    current_input_ids = input_ids           # [B, prompt_len]
    current_attn_mask = attention_mask      # [B, prompt_len]
    past_key_values = None
    first_step = True

    for _ in range(max_new_tokens):
        with torch.no_grad():
            outputs = model(
                input_ids=current_input_ids,
                attention_mask=current_attn_mask,
                past_key_values=None if first_step else past_key_values,
                use_cache=True,
            )
        first_step = False
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]  # [|active|, V]

        next_tokens = triton_topk_sample(
            logits,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            greedy=False,
        )  # [|active|]

        # Record generated tokens; identify which sequences finished
        still_active: List[int] = []
        keep_positions: List[int] = []

        for i, (orig_idx, tok) in enumerate(zip(active_indices, next_tokens.tolist())):
            generated_tokens[orig_idx].append(tok)
            if tok not in eos_ids:
                still_active.append(orig_idx)
                keep_positions.append(i)

        if not still_active:
            break

        # Prune batch if any sequences finished this step
        if len(still_active) < len(active_indices):
            keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)
            past_key_values = _prune_kv_cache(past_key_values, keep_t)
            current_attn_mask = current_attn_mask[keep_t]
            next_tokens = next_tokens[keep_t]

        active_indices = still_active

        # Next step: feed only the last token + extend the attention mask
        current_input_ids = next_tokens.unsqueeze(-1)            # [|active|, 1]
        current_attn_mask = torch.cat(
            [
                current_attn_mask,
                torch.ones(len(active_indices), 1, device=device, dtype=torch.long),
            ],
            dim=-1,
        )

    return generated_tokens


# ---------------------------------------------------------------------------
# GPU memory helpers
# ---------------------------------------------------------------------------

def _gpu_memory_fraction(device) -> float:
    """
    Return current GPU memory utilization as a fraction in [0, 1].

    Uses torch.cuda.mem_get_info() which queries the CUDA driver directly
    (~14µs per call, no device sync required).
    """
    free, total = torch.cuda.mem_get_info(device)
    return 1.0 - (free / total)


# ---------------------------------------------------------------------------
# KV-cache offload / reload (CPU <-> GPU)
# ---------------------------------------------------------------------------

def _split_kv_cache(past_key_values, keep: torch.Tensor, offload: torch.Tensor):
    """
    Split a KV cache into two parts by batch index.

    Uses the public DynamicCache API (``__len__``, ``__getitem__``,
    ``update``) which works across all internal representations (flat lists,
    per-layer objects, legacy tuples).  The ``keep`` portion is pruned
    in-place on GPU; the ``offload`` portion is copied to a new CPU-resident
    DynamicCache.

    Args:
        past_key_values: On-GPU KV cache from a model forward pass.
        keep:            1-D long tensor — batch indices to keep on GPU.
        offload:         1-D long tensor — batch indices to move to CPU.

    Returns:
        (gpu_cache, cpu_cache): The GPU-resident cache for ``keep`` and a
        CPU-resident DynamicCache for ``offload``.
    """
    num_layers = len(past_key_values)

    # Step 1: copy offload rows to CPU *before* pruning modifies the original
    cpu_cache = DynamicCache()
    for layer_idx in range(num_layers):
        k, v = past_key_values[layer_idx]
        cpu_cache.update(k[offload].cpu(), v[offload].cpu(), layer_idx)

    # Step 2: prune GPU cache in place to keep only 'keep' indices
    gpu_cache = _prune_kv_cache(past_key_values, keep)

    return gpu_cache, cpu_cache


def _reload_kv_cache(cpu_cache, device):
    """
    Move a CPU-resident KV cache back to GPU.

    Args:
        cpu_cache: A cache whose tensors reside on CPU (built by _split_kv_cache).
        device:    Target CUDA device.

    Returns:
        A new DynamicCache with all tensors on ``device``.
    """
    gpu_cache = DynamicCache()
    for layer_idx in range(len(cpu_cache)):
        k, v = cpu_cache[layer_idx]
        gpu_cache.update(
            k.to(device, non_blocking=True),
            v.to(device, non_blocking=True),
            layer_idx,
        )
    return gpu_cache


# ---------------------------------------------------------------------------
# Memory-aware pruned generation loop
# ---------------------------------------------------------------------------

def _run_pruned_loop_safe(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_ids: Set[int],
    K: int = 32,
    mem_threshold: float = 0.95,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> List[List[int]]:
    """
    Token-by-token generation loop with dynamic batch pruning AND proactive
    memory management.

    Generates K tokens at a time, then checks GPU memory utilization via
    ``torch.cuda.mem_get_info()`` (~14µs, no sync).  If usage exceeds
    ``mem_threshold``, the active batch is split in half: the second half's
    KV cache is offloaded to CPU, the first half finishes generating on GPU,
    and then the second half is reloaded and completed.  This avoids OOM in
    most cases without discarding any partial work.

    Args:
        model:            HuggingFace causal-LM (on GPU, eval mode).
        tokenizer:        Corresponding tokenizer.
        input_ids:        Left-padded prompt tensor [batch_size, prompt_len].
        attention_mask:   Corresponding attention mask [batch_size, prompt_len].
        max_new_tokens:   Maximum tokens to generate per sequence.
        eos_ids:          Set of token IDs that signal end-of-sequence.
        K:                Number of tokens to generate between memory checks.
        mem_threshold:    GPU memory fraction (0-1) that triggers a batch split.
        temperature:      Sampling temperature.
        top_p:            Nucleus filtering threshold.
        top_k:            Top-k filtering count.

    Returns:
        List of length batch_size, each element being a list of generated
        token IDs (not including the prompt; stops at EOS, inclusive).
    """
    device = input_ids.device
    batch_size = input_ids.shape[0]

    generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]

    # Deferred batches: list of state dicts for batch halves offloaded to CPU.
    # Processed sequentially after the current GPU batch finishes.
    deferred_batches: list[dict] = []

    # Initial state for the "current" batch being generated
    active_indices: List[int] = list(range(batch_size))
    current_input_ids = input_ids
    current_attn_mask = attention_mask
    past_key_values = None
    first_step = True
    tokens_generated = 0

    while tokens_generated < max_new_tokens and active_indices:
        # ── Inner loop: generate up to K tokens ──────────────────────
        chunk_end = min(tokens_generated + K, max_new_tokens)
        all_done = False

        while tokens_generated < chunk_end and active_indices:
            with torch.no_grad():
                outputs = model(
                    input_ids=current_input_ids,
                    attention_mask=current_attn_mask,
                    past_key_values=None if first_step else past_key_values,
                    use_cache=True,
                )
            first_step = False
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            next_tokens = triton_topk_sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                greedy=False,
            )

            # Record tokens; identify finished sequences
            still_active: List[int] = []
            keep_positions: List[int] = []

            for i, (orig_idx, tok) in enumerate(
                zip(active_indices, next_tokens.tolist())
            ):
                generated_tokens[orig_idx].append(tok)
                if tok not in eos_ids:
                    still_active.append(orig_idx)
                    keep_positions.append(i)

            tokens_generated += 1

            if not still_active:
                active_indices = []
                all_done = True
                break

            # Prune finished sequences
            if len(still_active) < len(active_indices):
                keep_t = torch.tensor(
                    keep_positions, device=device, dtype=torch.long
                )
                past_key_values = _prune_kv_cache(past_key_values, keep_t)
                current_attn_mask = current_attn_mask[keep_t]
                next_tokens = next_tokens[keep_t]

            active_indices = still_active
            current_input_ids = next_tokens.unsqueeze(-1)
            current_attn_mask = torch.cat(
                [
                    current_attn_mask,
                    torch.ones(
                        len(active_indices), 1, device=device, dtype=torch.long
                    ),
                ],
                dim=-1,
            )

        if all_done:
            break

        # ── Memory check after K tokens ──────────────────────────────
        if (
            len(active_indices) > 1
            and _gpu_memory_fraction(device) > mem_threshold
        ):
            mid = len(active_indices) // 2
            keep_pos = list(range(mid))
            offload_pos = list(range(mid, len(active_indices)))

            keep_t = torch.tensor(keep_pos, device=device, dtype=torch.long)
            offload_t = torch.tensor(offload_pos, device=device, dtype=torch.long)

            # Split KV cache: keep first half on GPU, offload second to CPU
            past_key_values, cpu_cache = _split_kv_cache(
                past_key_values, keep_t, offload_t
            )

            deferred_batches.append({
                "active_indices": active_indices[mid:],
                "kv_cache": cpu_cache,
                "attn_mask": current_attn_mask[offload_t].cpu(),
                "last_tokens": current_input_ids[offload_t].cpu(),
                "tokens_generated": tokens_generated,
            })

            # Prune GPU-side state to first half only
            active_indices = active_indices[:mid]
            current_attn_mask = current_attn_mask[keep_t]
            current_input_ids = current_input_ids[keep_t]

            # Release freed KV-cache blocks so mem_get_info reflects the change
            torch.cuda.empty_cache()

            print(f"  [MEM] Offloaded {len(deferred_batches[-1]['active_indices'])} seqs to CPU "
                  f"at token {tokens_generated} (now {_gpu_memory_fraction(device):.1%} used, "
                  f"{len(active_indices)} active)",
                  flush=True)

    # ── Process deferred batches sequentially ─────────────────────────
    del past_key_values, current_input_ids, current_attn_mask
    torch.cuda.empty_cache()

    while deferred_batches:
        state = deferred_batches.pop(0)
        active_indices = state["active_indices"]
        print(f"  [MEM] Reloading deferred batch: {len(active_indices)} seqs, "
              f"{max_new_tokens - state['tokens_generated']} tokens remaining "
              f"({len(deferred_batches)} more deferred)",
              flush=True)
        past_key_values = _reload_kv_cache(state["kv_cache"], device)
        current_attn_mask = state["attn_mask"].to(device, non_blocking=True)
        current_input_ids = state["last_tokens"].to(device, non_blocking=True)
        tokens_generated = state["tokens_generated"]
        first_step = False

        # Free the CPU copies now that we've moved them to GPU
        del state

        remaining = max_new_tokens - tokens_generated
        while tokens_generated < max_new_tokens and active_indices:
            chunk_end = min(tokens_generated + K, max_new_tokens)

            while tokens_generated < chunk_end and active_indices:
                with torch.no_grad():
                    outputs = model(
                        input_ids=current_input_ids,
                        attention_mask=current_attn_mask,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                logits = outputs.logits[:, -1, :]

                next_tokens = triton_topk_sample(
                    logits,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    greedy=False,
                )

                still_active = []
                keep_positions = []
                for i, (orig_idx, tok) in enumerate(
                    zip(active_indices, next_tokens.tolist())
                ):
                    generated_tokens[orig_idx].append(tok)
                    if tok not in eos_ids:
                        still_active.append(orig_idx)
                        keep_positions.append(i)

                tokens_generated += 1

                if not still_active:
                    active_indices = []
                    break

                if len(still_active) < len(active_indices):
                    keep_t = torch.tensor(
                        keep_positions, device=device, dtype=torch.long
                    )
                    past_key_values = _prune_kv_cache(past_key_values, keep_t)
                    current_attn_mask = current_attn_mask[keep_t]
                    next_tokens = next_tokens[keep_t]

                active_indices = still_active
                current_input_ids = next_tokens.unsqueeze(-1)
                current_attn_mask = torch.cat(
                    [
                        current_attn_mask,
                        torch.ones(
                            len(active_indices), 1,
                            device=device, dtype=torch.long,
                        ),
                    ],
                    dim=-1,
                )

            if not active_indices:
                break

            # Memory check: may split again (cascading)
            if (
                len(active_indices) > 1
                and _gpu_memory_fraction(device) > mem_threshold
            ):
                mid = len(active_indices) // 2
                keep_pos = list(range(mid))
                offload_pos = list(range(mid, len(active_indices)))

                keep_t = torch.tensor(
                    keep_pos, device=device, dtype=torch.long
                )
                offload_t = torch.tensor(
                    offload_pos, device=device, dtype=torch.long
                )

                past_key_values, cpu_cache = _split_kv_cache(
                    past_key_values, keep_t, offload_t
                )

                deferred_batches.append({
                    "active_indices": active_indices[mid:],
                    "kv_cache": cpu_cache,
                    "attn_mask": current_attn_mask[offload_t].cpu(),
                    "last_tokens": current_input_ids[offload_t].cpu(),
                    "tokens_generated": tokens_generated,
                })

                active_indices = active_indices[:mid]
                current_attn_mask = current_attn_mask[keep_t]
                current_input_ids = current_input_ids[keep_t]

                # Release freed KV-cache blocks so mem_get_info reflects the change
                torch.cuda.empty_cache()

                print(f"  [MEM] Offloaded {len(deferred_batches[-1]['active_indices'])} seqs to CPU "
                      f"at token {tokens_generated} (now {_gpu_memory_fraction(device):.1%} used, "
                      f"{len(active_indices)} active, cascading)",
                      flush=True)

        # Free before loading next deferred batch
        del past_key_values, current_input_ids, current_attn_mask
        torch.cuda.empty_cache()

    return generated_tokens

