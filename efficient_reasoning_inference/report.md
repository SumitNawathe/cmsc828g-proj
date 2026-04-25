# Efficient Batch Inference for Reasoning Models via Dynamic Sequence Offloading

## 1. Problem Statement

When serving a language model with a batch of N prompts, the standard
HuggingFace `model.generate()` API pads every sequence to the same length and
keeps all N sequences active until the *longest* one finishes.  For reasoning
workloads (e.g. MATH500 with chain-of-thought), output lengths are highly
variable — a straightforward arithmetic problem might terminate in 80 tokens
while a geometry proof requires 2000.  The result is significant waste:

- **Memory**: Every active sequence occupies a full row in the KV cache.
  Sequences that have already emitted an EOS token still hold
  `num_layers × 2 × seq_len × num_heads × head_dim` floats on the GPU.
- **Compute**: At every decode step, the attention kernel runs over all N rows
  of the KV cache, even for sequences whose outputs have already been written.

This report describes the efficient inference system implemented in
`src/efficient_inference.py` that eliminates this waste through
**dynamic sequence offloading**, and the custom Triton kernels in
`src/triton_kernels.py` that accelerate the per-step sampling operation.

---

## 2. Vanilla Baseline

The baseline (`src/vanilla_inference.py`) is a thin wrapper around
`model.generate()`.  All N prompts are left-padded to a common length, passed
as a single tensor, and the standard HuggingFace loop runs until every sequence
has either emitted EOS or exhausted `max_new_tokens`.

To allow a fair comparison of per-step resource usage, a `LogitsProcessor`
hook (`_StepLogger`) is injected into the generation loop.  It is called at
each decode step and records the current GPU memory allocation and how many
sequences have logically finished (emitted EOS) but are still being carried in
the batch.  This makes the "wasted compute" visible: the computed batch size
stays flat at N throughout, even though an increasing number of those rows
contribute nothing to the final outputs.

---

## 3. Efficient Inference: Dynamic Sequence Offloading

### 3.1 High-Level Algorithm

Instead of delegating generation to `model.generate()`, the efficient approach
implements a **manual token-by-token loop** that maintains the following mutable
state:

| Variable | Shape | Meaning |
|---|---|---|
| `active_indices` | `[A]` (Python list) | Which original prompt indices are still generating |
| `current_input_ids` | `[A, 1]` | The single new token for each active sequence |
| `current_attn_mask` | `[A, L]` | Cumulative attention mask (grows by 1 each step) |
| `past_key_values` | KV cache for A sequences | Accumulated keys/values for all past tokens |

At each step the loop performs four operations:

1. **Forward pass** — run one decode step over the active batch only.
2. **Sample** — draw the next token for each active sequence.
3. **Detect termination** — check which sampled tokens are EOS.
4. **Prune** — remove finished sequences and shrink all state tensors.

### 3.2 Prefill (Step 0)

On the first iteration, `past_key_values=None` is passed and
`current_input_ids` is the full padded prompt matrix `[N, prompt_len]`.  The
model processes all input tokens in parallel (prefill), returns logits and a
populated KV cache.  This is identical to what vanilla does; the difference
only emerges during the decode phase.

### 3.3 Decode Loop

From step 1 onward, `current_input_ids` has shape `[A, 1]` — just the most
recently sampled token for each of the A still-active sequences.  The model
uses the KV cache to reconstruct the full attention context without
reprocessing past tokens.

After sampling, every sequence whose new token is an EOS token is marked as
done.  Its index is removed from `active_indices` and its position is excluded
from `keep_positions`.

### 3.4 KV-Cache Pruning

This is the core of the efficiency gain.  When any sequence finishes, its row
must be removed from the KV cache so subsequent steps do not compute over it.

The pruning function `_prune_kv_cache(past_key_values, keep)` takes an integer
index tensor `keep` (the positions of still-active sequences in the current
batch) and slices every per-layer key and value tensor along dimension 0:

```
K[layer]  : [A, H, L, D]  →  K[layer][keep]  : [A', H, L, D]
V[layer]  : [A, H, L, D]  →  V[layer][keep]  : [A', H, L, D]
```

where `A' < A` is the new active batch size.  The function modifies the
existing cache object in-place and returns it unchanged so the model continues
to receive the same cache type it returned (important for Qwen3 which uses a
custom `DynamicCache` subclass with `layers`, `layer_class_to_replicate`, and
`offloading` attributes rather than the standard `key_cache`/`value_cache`
lists).  The pruner handles this through introspection: it scans the instance
`__dict__` for any lists of 4-D tensors `[batch, heads, seq, dim]` and slices
them regardless of their attribute name.

The attention mask is pruned in the same step:

```python
current_attn_mask = current_attn_mask[keep_t]   # [A', L]
```

and then extended by one column of ones for the new token:

```python
current_attn_mask = torch.cat(
    [current_attn_mask,
     torch.ones(A', 1, device=device, dtype=torch.long)],
    dim=-1
)   # [A', L+1]
```

The model automatically derives correct RoPE position IDs from the cumulative
sum of the attention mask, so no explicit `position_ids` manipulation is
required.

### 3.5 Memory and Compute Savings

The KV cache for a single sequence at sequence length L is:

```
2 × num_layers × num_heads × L × head_dim × sizeof(dtype)
```

For Qwen3-4B (28 layers, 8 KV heads, head_dim=128, bfloat16):

```
2 × 28 × 8 × L × 128 × 2 bytes  =  917 504 × L bytes  ≈  0.9 MB per token per sequence
```

When a sequence finishes at step t < max_new_tokens, every subsequent step
saves this entire allocation *and* the attention computation over it.  For a
batch of 8 sequences with variable lengths, the cumulative saving over the full
decode grows quadratically with the length difference between the shortest and
longest sequence.

---

## 4. Triton Kernels

### 4.1 Motivation

At each decode step, the sampling pipeline applies to logits of shape
`[A, V]` where `V ≈ 152 064` for Qwen3.  For greedy decoding this reduces to
a row-wise argmax; with sampling it involves temperature scaling, top-k
masking, top-p (nucleus) filtering, and multinomial draw.  These are all
memory-bandwidth-bound operations, and PyTorch's generic kernels may perform
unnecessary allocations (e.g. `torch.topk` with k=1 can still sort a
large prefix of the vocab).

### 4.2 Two-Pass Row-Wise Argmax (`triton_greedy_decode`)

The Triton greedy-decode kernel operates in two passes over the logit matrix,
launching one CUDA thread block (CTA) per batch element:

**Pass 1 — `_row_max_kernel`**

Each CTA sweeps its assigned row in chunks of `BLOCK_V = 1024` elements,
maintaining a scalar running maximum:

```python
running_max = tl.where(block_max > running_max, block_max, running_max)
```

The result is written to a temporary `[B]` float32 buffer.

**Pass 2 — `_row_argmax_kernel`**

Each CTA re-reads its row in the same chunked fashion.  For each chunk it
identifies positions where the value equals the global maximum found in pass 1,
replaces non-matching positions with the out-of-range sentinel `V`, then takes
the block minimum — yielding the index of the first occurrence of the maximum:

```python
cand = tl.where(is_max, offs, V)          # non-max → sentinel
block_min = tl.min(cand, axis=0)
best_idx = tl.where(block_min < best_idx, block_min, best_idx)
```

The two-pass structure avoids materialising a full sorted copy of the vocab and
keeps register pressure low.

### 4.3 Fused Sampling (`triton_topk_sample`)

For stochastic sampling the function fuses temperature scaling, top-k masking,
and top-p nucleus filtering into a single PyTorch call chain (no intermediate
full-vocab allocations), then dispatches to `torch.multinomial`.  When greedy
decoding is requested it delegates directly to `triton_greedy_decode`.  If
Triton is not installed, both functions fall back to equivalent PyTorch
implementations transparently.

---

## 5. Generation Configuration

Both methods use the Qwen3 default generation configuration:

| Parameter | Value |
|---|---|
| `do_sample` | `True` |
| `temperature` | 0.6 |
| `top_p` | 0.95 |
| `top_k` | 20 |
| EOS token IDs | `{151643, 151645}` (`<\|endoftext\|>`, `<\|im_end\|>`) |
| Chat template | Qwen3 with `enable_thinking=True` |

The `enable_thinking=True` flag causes the tokenizer to insert the thinking
preamble into the prompt, activating Qwen3's extended chain-of-thought mode.
This maximises output-length variance across problems — which is exactly the
regime where dynamic offloading yields the largest gains.

---

## 6. Summary of Differences

| Aspect | Vanilla | Efficient |
|---|---|---|
| Entry point | `model.generate()` | Manual decode loop |
| Batch size during decode | Always N | Shrinks as sequences finish |
| KV-cache rows | Always N | Drops to N' ≤ N at each EOS |
| Attention mask | Fixed length | Extended by 1 each step |
| Sampling kernel | HuggingFace built-in | Triton argmax / PyTorch sampling |
| Peak GPU memory | O(N × max_len) | O(N × avg_len) |
| Wall-clock time | Proportional to max output length | Proportional to avg output length |
