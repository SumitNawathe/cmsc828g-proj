# Efficient Reasoning Inference – Dynamic Sequence Offloading

Benchmark comparing **vanilla HuggingFace batch inference** against an
**efficient approach that dynamically removes finished sequences** from the
active batch during auto-regressive generation.

---

## The Problem

When generating N answers in parallel with `model.generate()`, every sequence
stays in the batch until the *longest* one finishes.  Sequences that finish
early are padded – they continue to:

- occupy **KV-cache rows** (GPU memory proportional to their sequence length),
- receive **full attention computation** at every subsequent step.

For reasoning tasks (e.g. MATH500 with chain-of-thought), output lengths vary
dramatically – some problems need only a short proof while others require
thousands of tokens.  This variance makes the wasted-compute problem severe.

---

## The Solution

Replace `model.generate()` with a **custom token-by-token loop** that:

1. Samples the next token for each active sequence.
2. Detects which sequences generated an EOS token.
3. **Drops those rows** from the batch: KV-cache is sliced, attention mask is
   trimmed, and the live batch shrinks.
4. Continues generating only for the remaining sequences.

A custom **Triton kernel** (`src/triton_kernels.py`) provides a fused
two-pass row-wise argmax over the vocabulary for greedy decoding, avoiding
redundant memory bandwidth compared to the generic PyTorch implementation.

---

## Project Structure

```
efficient_reasoning_inference/
├── benchmark.py          # Main benchmarking entry point
├── plot_results.py       # Generate figures from saved JSON results
├── requirements.txt
├── results/              # Auto-created; stores JSON + PNG/PDF plots
└── src/
    ├── data_utils.py     # MATH500 loading & Qwen3 prompt formatting
    ├── metrics.py        # BenchmarkResult dataclass, GPU memory helpers
    ├── triton_kernels.py # Triton two-pass argmax + fused sampling kernel
    ├── vanilla_inference.py   # Standard HF generate() baseline
    └── efficient_inference.py # Custom loop with dynamic sequence removal
```

---

## Quick Start

### 1 – Install dependencies
```bash
pip install -r requirements.txt
```

### 2 – Run the benchmark
```bash
python benchmark.py \
    --model Qwen/Qwen3-4B \
    --batch-sizes 4 8 16 \
    --n-problems 64 \
    --max-new-tokens 512 \
    --output-dir results/
```

A subset of MATH500 questions is used by default.  Adjust `--n-problems` and
`--max-new-tokens` to trade off runtime vs. statistical coverage.

### 3 – Generate figures
```bash
python plot_results.py --results results/benchmark_results.json
```

Figures are written to `results/`:

| File | Contents |
|---|---|
| `benchmark_comparison.png` | 6-panel overview: time, memory, throughput, speedup, active-batch size, memory over time |
| `active_sequences_detail.png` | Per-step active-sequence count (vanilla vs efficient) |
| `memory_detail.png` | Per-step GPU memory allocation |

---

## Generation Config (Qwen3 defaults)

| Parameter | Value |
|---|---|
| `do_sample` | `True` |
| `temperature` | `0.6` |
| `top_p` | `0.95` |
| `top_k` | `20` |
| `eos_token_id` | `[151643, 151645]` (Qwen3 EOS tokens) |
| `enable_thinking` | `True` (via chat template) |

Use `--greedy` for deterministic decoding (faster, no sampling variance).

---

## Expected Results

For reasoning workloads with high output-length variance, the efficient approach
typically shows:

- **1.3 – 2.5× wall-clock speedup** (larger at higher batch sizes)
- **10 – 40% lower peak GPU memory** (depends on sequence length distribution)
- **Higher effective throughput** (tokens/second) due to shrinking batch

---

## Implementation Notes

### KV-cache pruning
When a sequence finishes, its row in `past_key_values` is dropped with a
simple index-select:
```python
past_key_values = tuple((k[keep], v[keep]) for k, v in past_key_values)
```
This is an O(active_batch × num_layers × seq_len × head_dim) copy, which is
cheap compared to the attention computation it avoids.

### Triton kernel
`src/triton_kernels.py` implements a two-pass row-wise argmax:
- **Pass 1** (`_row_max_kernel`): one CTA per batch element; iterates over
  vocabulary in BLOCK_V=1024 chunks and keeps a running scalar maximum.
- **Pass 2** (`_row_argmax_kernel`): loads the same rows again, emits the
  first index whose value equals the row max.

This avoids materialising a full sorted copy of the 150K-token vocabulary that
`torch.topk(k=1)` may produce internally, and is fused with temperature
scaling in the sampling path.

If Triton is not installed, both functions fall back to pure PyTorch
automatically (`HAS_TRITON = False`).
