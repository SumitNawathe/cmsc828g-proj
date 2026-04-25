"""
benchmark.py – compare vanilla vs. efficient batch inference on MATH500.

Usage
-----
    python benchmark.py [OPTIONS]

Key options
-----------
  --model          HF model id              (default: Qwen/Qwen3-4B)
  --batch-sizes    space-separated ints     (default: 4 8 16)
  --n-problems     total problems to load   (default: 64)
  --max-new-tokens generation budget        (default: 512)
  --output-dir     where to save JSON+plots (default: results/)
  --seed           random seed              (default: 42)
  --dtype          bfloat16 | float16       (default: bfloat16)
  --greedy         use greedy decode        (flag, off by default)
  --no-triton      disable Triton kernel    (flag, off by default)
  --log-interval   steps between metric snapshots (default: 10)

Results are written to  <output-dir>/benchmark_results.json  and the
summary table is printed to stdout.  Run plot_results.py afterwards to
generate the comparison figures.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.data_utils import get_prompts
from src.efficient_inference import EfficientInference
from src.triton_kernels import HAS_TRITON
from src.vanilla_inference import VanillaInference


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(model_name: str, dtype_str: str):
    print(f"[benchmark] Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if dtype_str == "bfloat16" else torch.float16
    print(f"[benchmark] Loading model (dtype={dtype}) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="cuda",
        trust_remote_code=True,
    )
    model.eval()
    print(f"[benchmark] Model loaded on {next(model.parameters()).device}")
    return model, tokenizer


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _sep(char: str = "─", width: int = 70) -> str:
    return char * width


def _print_result(tag: str, r) -> None:
    print(
        f"  {tag:<22}  "
        f"time={r.total_time_s:6.1f}s  "
        f"peak_mem={r.peak_memory_mb/1024:5.2f}GB  "
        f"tps={r.tokens_per_second:7.1f}  "
        f"tokens={r.total_tokens:6d}"
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    model_name: str,
    batch_sizes: List[int],
    n_problems: int,
    max_new_tokens: int,
    output_dir: str,
    seed: int,
    dtype_str: str,
    greedy: bool,
    use_triton: bool,
    log_interval: int,
    temperature: float = 0.6,
    top_p: float = 0.95,
    top_k: int = 20,
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(model_name, dtype_str)

    total_needed = max(batch_sizes)
    n_load = max(n_problems, total_needed)
    print(f"[benchmark] Loading {n_load} MATH500 prompts (seed={seed}) ...")
    prompts, _ = get_prompts(n_load, tokenizer, seed=seed)
    print(f"[benchmark] Loaded {len(prompts)} prompts.")

    vanilla_runner = VanillaInference(model, tokenizer)
    # Custom loop with PyTorch sampling (isolates KV-cache pruning contribution)
    efficient_pt_runner = EfficientInference(
        model, tokenizer, use_triton=False, greedy=greedy
    )
    # Custom loop with Triton sampling (KV-cache pruning + Triton kernel)
    efficient_triton_runner = EfficientInference(
        model, tokenizer, use_triton=use_triton, greedy=greedy
    )

    all_results = []

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        log_interval=log_interval,
    )

    print(f"\n[benchmark] Triton available: {HAS_TRITON}  |  using Triton: {use_triton and HAS_TRITON}")
    print(f"[benchmark] Greedy decode: {greedy}")
    print(f"[benchmark] max_new_tokens={max_new_tokens}, temperature={temperature}, "
          f"top_p={top_p}, top_k={top_k}\n")

    for bs in batch_sizes:
        batch_prompts = prompts[:bs]

        print(_sep())
        print(f"  Batch size = {bs}")
        print(_sep())

        # ---- Vanilla ----
        print("  [1/3] Vanilla HF generate() ...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.perf_counter()
        v_res = vanilla_runner.run_batch(batch_prompts, **gen_kwargs)
        print(f"        done in {time.perf_counter()-t:.1f}s")
        _print_result("vanilla", v_res)
        all_results.append(v_res.to_dict())

        # ---- Efficient + PyTorch sampling (no Triton) ----
        print("  [2/3] Efficient + PyTorch sampling (pruning only, no Triton) ...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.perf_counter()
        ep_res = efficient_pt_runner.run_batch(batch_prompts, **gen_kwargs)
        ep_res.method = "efficient_pt"   # rename so JSON is distinguishable
        print(f"        done in {time.perf_counter()-t:.1f}s")
        _print_result("efficient_pt", ep_res)
        all_results.append(ep_res.to_dict())

        # ---- Efficient + Triton sampling ----
        print("  [3/3] Efficient + Triton sampling (pruning + Triton kernel) ...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        t = time.perf_counter()
        e_res = efficient_triton_runner.run_batch(batch_prompts, **gen_kwargs)
        print(f"        done in {time.perf_counter()-t:.1f}s")
        _print_result("efficient", e_res)
        all_results.append(e_res.to_dict())

        def _pct(a, b):
            return (a - b) / max(b, 1e-6) * 100

        print(
            f"\n  ► Pruning speedup   (vanilla → efficient_pt):     "
            f"{v_res.total_time_s / max(ep_res.total_time_s, 1e-6):.2f}×  "
            f"({_pct(ep_res.tokens_per_second, v_res.tokens_per_second):+.1f}% tps)"
        )
        print(
            f"  ► Triton kernel gain (efficient_pt → efficient):   "
            f"{ep_res.total_time_s / max(e_res.total_time_s, 1e-6):.2f}×  "
            f"({_pct(e_res.tokens_per_second, ep_res.tokens_per_second):+.1f}% tps)"
        )
        print(
            f"  ► Combined speedup  (vanilla → efficient):         "
            f"{v_res.total_time_s / max(e_res.total_time_s, 1e-6):.2f}×  "
            f"({_pct(e_res.tokens_per_second, v_res.tokens_per_second):+.1f}% tps)\n"
        )

    # Save
    out_path = os.path.join(output_dir, "benchmark_results.json")
    with open(out_path, "w") as fh:
        json.dump(all_results, fh, indent=2)
    print(_sep("="))
    print(f"  Results saved → {out_path}")
    print(f"  Run `python plot_results.py` to generate figures.")
    print(_sep("="))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-4B",
                   help="HuggingFace model id (default: Qwen/Qwen3-4B)")
    p.add_argument("--batch-sizes", nargs="+", type=int, default=[4, 8, 16],
                   metavar="N", help="Batch sizes to benchmark (default: 4 8 16)")
    p.add_argument("--n-problems", type=int, default=64,
                   help="Number of MATH500 problems to load (default: 64)")
    p.add_argument("--max-new-tokens", type=int, default=512,
                   help="Max tokens to generate per sequence (default: 512)")
    p.add_argument("--output-dir", default="results",
                   help="Directory for JSON results and plots (default: results/)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--log-interval", type=int, default=10,
                   help="Steps between per-step metric snapshots (default: 10)")
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy (argmax) decoding instead of sampling")
    p.add_argument("--no-triton", action="store_true",
                   help="Disable Triton kernel, use PyTorch sampling instead")
    return p.parse_args()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[benchmark] ERROR: CUDA is required.", file=sys.stderr)
        sys.exit(1)

    args = _parse_args()
    run_benchmark(
        model_name=args.model,
        batch_sizes=args.batch_sizes,
        n_problems=args.n_problems,
        max_new_tokens=args.max_new_tokens,
        output_dir=args.output_dir,
        seed=args.seed,
        dtype_str=args.dtype,
        greedy=args.greedy,
        use_triton=not args.no_triton,
        log_interval=args.log_interval,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
