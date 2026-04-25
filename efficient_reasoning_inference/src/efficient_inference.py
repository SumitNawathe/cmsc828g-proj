"""
Efficient batch inference with dynamic sequence offloading.

Key idea
--------
Standard `model.generate()` keeps every sequence in the batch alive until the
*longest* one finishes.  Sequences that hit an EOS token early are just padded
– they waste both GPU memory (their KV-cache rows are retained) and compute
(attention still runs over them).

This module replaces that behaviour with a **custom token-by-token loop** that:
  1. Samples the next token for every active sequence.
  2. Detects which sequences generated an EOS token.
  3. *Removes* those sequences from the active batch: their KV-cache rows are
     dropped in place, the attention mask is sliced, and the live batch shrinks.
  4. Continues only over the remaining (still active) sequences.

Sampling uses the Triton kernels from triton_kernels.py when available;
falls back to pure PyTorch otherwise.  The generation hyper-parameters follow
the Qwen3 defaults: temperature=0.6, top_p=0.95, top_k=20.
"""
from __future__ import annotations

import time
from typing import List, Set

import torch
from transformers import DynamicCache

from .metrics import (
    BenchmarkResult,
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_peak_memory,
)
from .triton_kernels import HAS_TRITON, triton_topk_sample

QWEN3_EOS_IDS: List[int] = [151643, 151645]


# ---------------------------------------------------------------------------
# KV-cache pruning helper
# ---------------------------------------------------------------------------

def _prune_kv_cache(past_key_values, keep: torch.Tensor):
    """
    Prune the batch dimension of a HuggingFace KV-cache object.

    We do NOT rely on specific attribute names (key_cache / value_cache)
    because those names differ across transformers versions.  Instead we
    inspect the object's instance __dict__ for any list of 4-D tensors
    (the universal [batch, heads, seq_len, head_dim] KV format) and slice
    them in-place, then return the same object so the model sees its own
    cache type (DynamicCache, HybridCache, etc.) rather than a plain tuple.

    Falls back to a fresh DynamicCache built via update() for legacy
    tuple-of-tuples caches.
    """
    def _prune_4d_tensors(obj) -> bool:
        """
        Find every 4-D tensor in obj's instance __dict__ and index-select
        along dim-0 (batch).  Returns True if at least one tensor was found.
        """
        found = False
        for name, val in list(vars(obj).items()):
            if isinstance(val, torch.Tensor) and val.ndim == 4:
                setattr(obj, name, val[keep])
                found = True
        return found

    # ------------------------------------------------------------------
    # Strategy 1a: flat lists of 4-D tensors  (e.g. key_cache / value_cache).
    #   Works with standard DynamicCache in transformers ≥4.38.
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Strategy 1b: per-layer cache objects stored in a 'layers' list.
    #   e.g. DynamicCache(layers=[LayerCache(...), ...])
    #   Each LayerCache holds 4-D key/value tensors directly.
    # ------------------------------------------------------------------
    layers_attr = getattr(past_key_values, "layers", None)
    if isinstance(layers_attr, list) and layers_attr:
        pruned_any = False
        for layer_cache in layers_attr:
            pruned_any |= _prune_4d_tensors(layer_cache)
            # Also probe one level deeper (some impls nest further)
            for name, val in list(vars(layer_cache).items()):
                if (isinstance(val, list) and val
                        and isinstance(val[0], torch.Tensor) and val[0].ndim == 4):
                    for i in range(len(val)):
                        val[i] = val[i][keep]
                    pruned_any = True
        if pruned_any:
            return past_key_values

    # ------------------------------------------------------------------
    # Strategy 2: legacy tuple-of-tuples  (transformers < 4.38).
    # ------------------------------------------------------------------
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
# EfficientInference
# ---------------------------------------------------------------------------

class EfficientInference:
    """
    Batch inference that dynamically removes finished sequences.

    Parameters
    ----------
    model       : HuggingFace causal-LM model (already on GPU, eval mode).
    tokenizer   : Corresponding tokenizer (padding_side will be set to "left").
    use_triton  : Use the Triton greedy/sampling kernel if available.
    greedy      : If True, use greedy (argmax) decoding instead of sampling.
    """

    def __init__(self, model, tokenizer, use_triton: bool = True, greedy: bool = False) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.use_triton = use_triton and HAS_TRITON
        self.greedy = greedy

        eos: Set[int] = set(QWEN3_EOS_IDS)
        if tokenizer.eos_token_id is not None:
            eos.add(int(tokenizer.eos_token_id))
        self.eos_ids = eos

    # ------------------------------------------------------------------
    # Internal sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
        if self.use_triton:
            return triton_topk_sample(
                logits,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                greedy=self.greedy,
            )
        # Pure-PyTorch fallback
        if self.greedy or temperature == 0.0:
            return torch.argmax(logits, dim=-1)
        scaled = logits / temperature if temperature != 1.0 else logits
        if top_k > 0:
            k = min(top_k, scaled.size(-1))
            threshold = torch.topk(scaled, k, dim=-1).values[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))
        if top_p < 1.0:
            sl, si = torch.sort(scaled, descending=True, dim=-1)
            cp = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
            sl = sl.masked_fill((cp - torch.softmax(sl, dim=-1)) > top_p, float("-inf"))
            scaled = scaled.scatter(1, si, sl)
        return torch.multinomial(torch.softmax(scaled, dim=-1), num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        log_interval: int = 10,
    ) -> BenchmarkResult:
        """
        Run the efficient custom generation loop and return metrics.

        The loop maintains:
          - `active_indices`      : which original prompt indices are still live
          - `current_input_ids`   : shape [|active|, 1] (or [|active|, prompt_len] on step 0)
          - `current_attn_mask`   : shape [|active|, total_seq_len_so_far]
          - `past_key_values`     : KV cache for active sequences only
        """
        device = next(self.model.parameters()).device
        batch_size = len(prompts)

        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        reset_peak_memory()
        torch.cuda.synchronize()

        # State
        active_indices: List[int] = list(range(batch_size))
        generated_tokens: List[List[int]] = [[] for _ in range(batch_size)]

        current_input_ids = inputs.input_ids          # [B, prompt_len]
        current_attn_mask = inputs.attention_mask     # [B, prompt_len]
        past_key_values = None
        first_step = True

        memory_log: List[float] = []
        computed_log: List[int] = []

        t0 = time.perf_counter()

        for step in range(max_new_tokens):
            # ----------------------------------------------------------
            # Forward pass
            # ----------------------------------------------------------
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_input_ids,
                    attention_mask=current_attn_mask,
                    past_key_values=None if first_step else past_key_values,
                    use_cache=True,
                )
            first_step = False
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]  # [|active|, V]

            # ----------------------------------------------------------
            # Sample next tokens
            # ----------------------------------------------------------
            next_tokens = self._sample(logits, temperature, top_p, top_k)  # [|active|]

            # ----------------------------------------------------------
            # Determine which sequences finished
            # ----------------------------------------------------------
            still_active: List[int] = []
            keep_positions: List[int] = []

            for i, (orig_idx, tok) in enumerate(zip(active_indices, next_tokens.tolist())):
                generated_tokens[orig_idx].append(tok)
                if tok not in self.eos_ids:
                    still_active.append(orig_idx)
                    keep_positions.append(i)

            # ----------------------------------------------------------
            # Step-level logging
            # ----------------------------------------------------------
            if step > 0 and step % log_interval == 0:
                memory_log.append(get_gpu_memory_mb())
                computed_log.append(len(active_indices))

            # All done?
            if not still_active:
                break

            # ----------------------------------------------------------
            # Prune batch if any sequences finished this step
            # ----------------------------------------------------------
            if len(still_active) < len(active_indices):
                keep_t = torch.tensor(keep_positions, device=device, dtype=torch.long)
                past_key_values = _prune_kv_cache(past_key_values, keep_t)
                current_attn_mask = current_attn_mask[keep_t]
                next_tokens = next_tokens[keep_t]

            active_indices = still_active

            # ----------------------------------------------------------
            # Prepare next-step inputs: single new token + extended mask
            # ----------------------------------------------------------
            current_input_ids = next_tokens.unsqueeze(-1)            # [|active|, 1]
            current_attn_mask = torch.cat(
                [
                    current_attn_mask,
                    torch.ones(len(active_indices), 1, device=device, dtype=torch.long),
                ],
                dim=-1,
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mem = get_peak_gpu_memory_mb()
        total_tokens = sum(len(t) for t in generated_tokens)

        return BenchmarkResult(
            method="efficient",
            batch_size=batch_size,
            total_time_s=elapsed,
            peak_memory_mb=peak_mem,
            tokens_per_second=total_tokens / elapsed,
            total_tokens=total_tokens,
            memory_over_time=memory_log,
            computed_seqs_over_time=computed_log,
            finished_seqs_over_time=[],   # tracked differently (via active drop)
        )
