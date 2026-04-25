"""
plot_results.py – generate comparison figures from benchmark_results.json.

Usage
-----
    python plot_results.py [--results results/benchmark_results.json]
                           [--output-dir results/]

Produces
--------
  benchmark_comparison.png / .pdf  – 6-panel overview figure
  active_sequences_detail.png      – per-step sequence-count comparison
  memory_detail.png                – per-step memory usage comparison

All figures use a clean two-colour scheme:
  Vanilla  →  coral / red
  Efficient →  green
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")   # headless-safe backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "#F8F9FA",
        "grid.color": "white",
        "grid.linewidth": 1.2,
    }
)

C_VANILLA      = "#E05252"   # warm red
C_EFFICIENT_PT = "#2980B9"   # blue  (pruning only, PyTorch sampling)
C_EFFICIENT    = "#27AE60"   # green (pruning + Triton kernel)
C_SAVED        = "#A9DFBF"   # light green fill
ALPHA          = 0.85


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load(path: str) -> List[dict]:
    with open(path) as fh:
        return json.load(fh)


def split(results: List[dict]):
    """
    Split results into per-method {batch_size: record} dicts.

    Supports both the old two-method format (vanilla / efficient) and the new
    three-method format (vanilla / efficient_pt / efficient).
    """
    vanilla: Dict[int, dict] = {}
    efficient_pt: Dict[int, dict] = {}
    efficient: Dict[int, dict] = {}
    for r in results:
        bs = r["batch_size"]
        m  = r["method"]
        if m == "vanilla":
            vanilla[bs] = r
        elif m == "efficient_pt":
            efficient_pt[bs] = r
        else:
            efficient[bs] = r
    # If no efficient_pt records exist (old-format file), leave dict empty
    batch_sizes = sorted(set(vanilla) | set(efficient_pt) | set(efficient))
    return vanilla, efficient_pt, efficient, batch_sizes


def _get(d: Dict[int, dict], key: str, default=0):
    return {bs: d.get(bs, {}).get(key, default) for bs in d}


# ---------------------------------------------------------------------------
# Individual panel functions
# ---------------------------------------------------------------------------

def _bars3(ax, batch_sizes, v_vals, ep_vals, e_vals, ylabel, title):
    """Helper: grouped bar chart for three series."""
    x = np.arange(len(batch_sizes))
    w = 0.25
    ax.bar(x - w, v_vals,  w, color=C_VANILLA,      alpha=ALPHA, label="Vanilla HF")
    ax.bar(x,     ep_vals, w, color=C_EFFICIENT_PT, alpha=ALPHA, label="Pruning only (PyTorch)")
    ax.bar(x + w, e_vals,  w, color=C_EFFICIENT,    alpha=ALPHA, label="Pruning + Triton")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.legend(fontsize=8); ax.grid(axis="y")


def panel_time(vanilla, efficient_pt, efficient, batch_sizes, ax):
    v_t  = [vanilla.get(bs, {}).get("total_time_s", 0) for bs in batch_sizes]
    ep_t = [efficient_pt.get(bs, {}).get("total_time_s", 0) for bs in batch_sizes]
    e_t  = [efficient.get(bs, {}).get("total_time_s", 0) for bs in batch_sizes]
    _bars3(ax, batch_sizes, v_t, ep_t, e_t, "Wall-clock time (s)", "Inference Time")


def panel_memory(vanilla, efficient_pt, efficient, batch_sizes, ax):
    v_m  = [vanilla.get(bs, {}).get("peak_memory_mb", 0) / 1024 for bs in batch_sizes]
    ep_m = [efficient_pt.get(bs, {}).get("peak_memory_mb", 0) / 1024 for bs in batch_sizes]
    e_m  = [efficient.get(bs, {}).get("peak_memory_mb", 0) / 1024 for bs in batch_sizes]
    _bars3(ax, batch_sizes, v_m, ep_m, e_m, "Peak GPU memory (GB)", "Peak GPU Memory")


def panel_throughput(vanilla, efficient_pt, efficient, batch_sizes, ax):
    v_tps  = [vanilla.get(bs, {}).get("tokens_per_second", 0) for bs in batch_sizes]
    ep_tps = [efficient_pt.get(bs, {}).get("tokens_per_second", 0) for bs in batch_sizes]
    e_tps  = [efficient.get(bs, {}).get("tokens_per_second", 0) for bs in batch_sizes]
    _bars3(ax, batch_sizes, v_tps, ep_tps, e_tps, "Tokens / second", "Throughput")


def panel_speedup(vanilla, efficient_pt, efficient, batch_sizes, ax):
    """
    Stacked speedup decomposition:
      - Blue bar  : gain from KV-cache pruning alone  (vanilla → efficient_pt)
      - Green bar : additional gain from Triton kernel (efficient_pt → efficient)
    """
    x = np.arange(len(batch_sizes))
    w = 0.5

    pruning_speedup = [
        vanilla.get(bs, {}).get("total_time_s", 1.0)
        / max(efficient_pt.get(bs, {}).get("total_time_s", 1.0), 1e-9)
        for bs in batch_sizes
    ]
    # Extra factor from Triton on top of pruning (relative to efficient_pt)
    triton_extra = [
        max(efficient_pt.get(bs, {}).get("total_time_s", 1.0), 1e-9)
        / max(efficient.get(bs, {}).get("total_time_s", 1.0), 1e-9)
        for bs in batch_sizes
    ]
    combined = [
        vanilla.get(bs, {}).get("total_time_s", 1.0)
        / max(efficient.get(bs, {}).get("total_time_s", 1.0), 1e-9)
        for bs in batch_sizes
    ]

    # Only show decomposed bars when efficient_pt data exists
    has_pt = any(efficient_pt)
    if has_pt:
        ax.bar(x, pruning_speedup, w, color=C_EFFICIENT_PT, alpha=ALPHA,
               label="KV-cache pruning gain")
        ax.bar(x, [t - 1.0 for t in triton_extra], w, bottom=pruning_speedup,
               color=C_EFFICIENT, alpha=ALPHA, label="Triton kernel gain")
    else:
        ax.bar(x, combined, w, color=C_EFFICIENT, alpha=ALPHA, label="Combined speedup")

    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="Baseline (1×)")
    for xi, s in zip(x, combined):
        ax.text(xi, s + 0.02, f"{s:.2f}×", ha="center", va="bottom",
                fontsize=10, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel("Speedup over vanilla")
    ax.set_title("Speedup Decomposition"); ax.legend(fontsize=8); ax.grid(axis="y")


def panel_active_seqs(vanilla, efficient_pt, efficient, batch_sizes, ax, log_interval: int = 10):
    """Active batch size over generation steps."""
    max_bs = max(batch_sizes)
    e_data  = efficient.get(max_bs, {})
    v_data  = vanilla.get(max_bs, {})

    eff_seqs    = e_data.get("computed_seqs_over_time", [])
    van_finished = v_data.get("finished_seqs_over_time", [])

    if not eff_seqs:
        ax.text(0.5, 0.5, "No step data\n(run benchmark first)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Active Sequences Over Time")
        return

    steps   = [i * log_interval for i in range(len(eff_seqs))]
    n_steps = len(eff_seqs)
    van_line = [max_bs] * n_steps

    ax.step(steps, van_line, where="post",
            color=C_VANILLA, lw=2.0, ls="--", label=f"Vanilla HF (always {max_bs})")
    ax.step(steps, eff_seqs, where="post",
            color=C_EFFICIENT, lw=2.0, label="Efficient (pruning + Triton)")
    ax.fill_between(steps, eff_seqs, van_line,
                    step="post", alpha=0.18, color=C_SAVED, label="Wasted compute (vanilla)")

    if van_finished:
        ax.step(steps[:len(van_finished)], van_finished, where="post",
                color=C_VANILLA, lw=1.2, ls=":", alpha=0.7,
                label="Vanilla: logically done (padded)")

    ax.set_xlabel(f"Generation step (sampled every {log_interval} steps)")
    ax.set_ylabel("Sequences in active batch")
    ax.set_title(f"Active-Batch Size Over Time  (B={max_bs})")
    ax.legend(fontsize=8); ax.grid(axis="y")
    ax.set_ylim(0, max_bs + 1)


def panel_memory_over_time(vanilla, efficient_pt, efficient, batch_sizes, ax, log_interval: int = 10):
    """Per-step GPU memory for all three methods."""
    max_bs = max(batch_sizes)

    def _mem(d):
        return [m / 1024 for m in d.get(max_bs, {}).get("memory_over_time", [])]

    van_mem = _mem(vanilla)
    ep_mem  = _mem(efficient_pt)
    eff_mem = _mem(efficient)

    if not van_mem and not eff_mem:
        ax.text(0.5, 0.5, "No memory data\n(run benchmark first)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("GPU Memory Over Time")
        return

    if van_mem:
        ax.plot([i * log_interval for i in range(len(van_mem))], van_mem,
                color=C_VANILLA, lw=2, ls="--", label="Vanilla HF")
    if ep_mem:
        ax.plot([i * log_interval for i in range(len(ep_mem))], ep_mem,
                color=C_EFFICIENT_PT, lw=2, ls="-.", label="Pruning only (PyTorch)")
    if eff_mem:
        ax.plot([i * log_interval for i in range(len(eff_mem))], eff_mem,
                color=C_EFFICIENT, lw=2, label="Pruning + Triton")

    ax.set_xlabel(f"Generation step (sampled every {log_interval} steps)")
    ax.set_ylabel("GPU memory allocated (GB)")
    ax.set_title(f"GPU Memory During Generation  (B={max_bs})")
    ax.legend(fontsize=9); ax.grid()


# ---------------------------------------------------------------------------
# Main composite figure
# ---------------------------------------------------------------------------

def make_overview_figure(results: List[dict], output_dir: str, log_interval: int) -> None:
    vanilla, efficient_pt, efficient, batch_sizes = split(results)

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    panel_time(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[0, 0]))
    panel_memory(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[0, 1]))
    panel_throughput(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[0, 2]))
    panel_speedup(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[1, 0]))
    panel_active_seqs(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[1, 1]), log_interval)
    panel_memory_over_time(vanilla, efficient_pt, efficient, batch_sizes, fig.add_subplot(gs[1, 2]), log_interval)

    fig.suptitle(
        "Dynamic Sequence Offloading: Efficient vs. Vanilla Batch Inference\n"
        "Model: Qwen3-4B  ·  Dataset: MATH500  ·  Qwen3 default generation config",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"benchmark_comparison.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {path}")
    plt.close(fig)


def make_active_seqs_figure(results: List[dict], output_dir: str, log_interval: int) -> None:
    vanilla, efficient_pt, efficient, batch_sizes = split(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    panel_active_seqs(vanilla, efficient_pt, efficient, batch_sizes, ax, log_interval)
    fig.tight_layout()

    path = os.path.join(output_dir, "active_sequences_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


def make_memory_figure(results: List[dict], output_dir: str, log_interval: int) -> None:
    vanilla, efficient_pt, efficient, batch_sizes = split(results)

    fig, ax = plt.subplots(figsize=(10, 5))
    panel_memory_over_time(vanilla, efficient_pt, efficient, batch_sizes, ax, log_interval)
    fig.tight_layout()

    path = os.path.join(output_dir, "memory_detail.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[plot] Saved → {path}")
    plt.close(fig)


def make_pruning_comparison_figure(results: List[dict], output_dir: str, log_interval: int) -> None:
    """
    6-panel figure comparing vanilla vs. pruning-only (no Triton).
    Same layout as benchmark_comparison.png; isolates the pruning contribution.
    """
    vanilla, efficient_pt, _, batch_sizes = split(results)

    if not efficient_pt:
        print("[plot] No 'efficient_pt' records found – re-run benchmark.py to get three-way data.")
        return

    max_bs = max(batch_sizes)
    x = np.arange(len(batch_sizes))
    w = 0.35

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)
    # fig.suptitle(
    #     "KV-Cache Pruning vs. Vanilla  (no Triton kernel)\n"
    #     "Model: Qwen3-4B  ·  Dataset: MATH500  ·  Qwen3 default generation config",
    #     fontsize=14, fontweight="bold", y=1.01,
    # )

    # ---- [0,0] Inference Time ----
    ax = fig.add_subplot(gs[0, 0])
    v_t  = [vanilla.get(bs, {}).get("total_time_s", 0) for bs in batch_sizes]
    ep_t = [efficient_pt.get(bs, {}).get("total_time_s", 0) for bs in batch_sizes]
    b1 = ax.bar(x - w/2, v_t,  w, color=C_VANILLA,      alpha=ALPHA, label="Vanilla HF")
    b2 = ax.bar(x + w/2, ep_t, w, color=C_EFFICIENT_PT, alpha=ALPHA, label="KV-cache Pruning")
    for b in list(b1) + list(b2):
        h = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, h + max(v_t)*0.01,
                f"{h:.1f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel("Wall-clock time (s)")
    ax.set_title("Inference Time"); ax.legend(fontsize=9); ax.grid(axis="y")

    # ---- [0,1] Peak Memory ----
    ax = fig.add_subplot(gs[0, 1])
    v_m  = [vanilla.get(bs, {}).get("peak_memory_mb", 0) / 1024 for bs in batch_sizes]
    ep_m = [efficient_pt.get(bs, {}).get("peak_memory_mb", 0) / 1024 for bs in batch_sizes]
    ax.bar(x - w/2, v_m,  w, color=C_VANILLA,      alpha=ALPHA, label="Vanilla HF")
    ax.bar(x + w/2, ep_m, w, color=C_EFFICIENT_PT, alpha=ALPHA, label="KV-cache Pruning")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel("Peak GPU memory (GB)")
    ax.set_title("Peak GPU Memory"); ax.legend(fontsize=9); ax.grid(axis="y")

    # ---- [0,2] Throughput ----
    ax = fig.add_subplot(gs[0, 2])
    v_tps  = [vanilla.get(bs, {}).get("tokens_per_second", 0) for bs in batch_sizes]
    ep_tps = [efficient_pt.get(bs, {}).get("tokens_per_second", 0) for bs in batch_sizes]
    ax.bar(x - w/2, v_tps,  w, color=C_VANILLA,      alpha=ALPHA, label="Vanilla HF")
    ax.bar(x + w/2, ep_tps, w, color=C_EFFICIENT_PT, alpha=ALPHA, label="KV-cache Pruning")
    ax.set_xticks(x); ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel("Tokens / second")
    ax.set_title("Throughput"); ax.legend(fontsize=9); ax.grid(axis="y")

    # ---- [1,0] Speedup ----
    ax = fig.add_subplot(gs[1, 0])
    speedups = [
        vanilla.get(bs, {}).get("total_time_s", 1.0)
        / max(efficient_pt.get(bs, {}).get("total_time_s", 1.0), 1e-9)
        for bs in batch_sizes
    ]
    colors = [C_EFFICIENT_PT if s >= 1 else C_VANILLA for s in speedups]
    bars = ax.bar(range(len(batch_sizes)), speedups, color=colors, alpha=ALPHA)
    ax.axhline(1.0, color="black", lw=1.2, ls="--", label="No speedup (1×)")
    for bar, s in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(batch_sizes)))
    ax.set_xticklabels([str(b) for b in batch_sizes])
    ax.set_xlabel("Batch size"); ax.set_ylabel("Speedup (vanilla / KV-cache pruning)")
    ax.set_title("Speedup from KV-Cache Pruning")
    ax.legend(fontsize=9); ax.grid(axis="y")

    # ---- [1,1] Active Sequences Over Time ----
    ax = fig.add_subplot(gs[1, 1])
    ep_seqs = efficient_pt.get(max_bs, {}).get("computed_seqs_over_time", [])
    van_fin = vanilla.get(max_bs, {}).get("finished_seqs_over_time", [])
    if ep_seqs:
        steps    = [i * log_interval for i in range(len(ep_seqs))]
        van_line = [max_bs] * len(ep_seqs)
        ax.step(steps, van_line, where="post",
                color=C_VANILLA, lw=2, ls="--", label=f"Vanilla HF (always {max_bs})")
        ax.step(steps, ep_seqs, where="post",
                color=C_EFFICIENT_PT, lw=2, label="KV-cachePruning")
        ax.fill_between(steps, ep_seqs, van_line,
                        step="post", alpha=0.18, color="#AED6F1",
                        label="Wasted compute (vanilla)")
        if van_fin:
            ax.step(steps[:len(van_fin)], van_fin, where="post",
                    color=C_VANILLA, lw=1.2, ls=":", alpha=0.7,
                    label="Vanilla: logically done (padded)")
        ax.set_ylim(0, max_bs + 1)
    else:
        ax.text(0.5, 0.5, "No step data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"Generation step (sampled every {log_interval} steps)")
    ax.set_ylabel("Sequences in active batch")
    ax.set_title(f"Active-Batch Size Over Time  (B={max_bs})")
    ax.legend(fontsize=8); ax.grid(axis="y")

    # ---- [1,2] Memory Over Time ----
    ax = fig.add_subplot(gs[1, 2])
    van_mem = [m / 1024 for m in vanilla.get(max_bs, {}).get("memory_over_time", [])]
    ep_mem  = [m / 1024 for m in efficient_pt.get(max_bs, {}).get("memory_over_time", [])]
    if van_mem:
        ax.plot([i * log_interval for i in range(len(van_mem))], van_mem,
                color=C_VANILLA, lw=2, ls="--", label="Vanilla HF")
    if ep_mem:
        ax.plot([i * log_interval for i in range(len(ep_mem))], ep_mem,
                color=C_EFFICIENT_PT, lw=2, label="KV-cache Pruning")
    if not van_mem and not ep_mem:
        ax.text(0.5, 0.5, "No memory data", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel(f"Generation step (sampled every {log_interval} steps)")
    ax.set_ylabel("GPU memory allocated (GB)")
    ax.set_title(f"GPU Memory During Generation  (B={max_bs})")
    ax.legend(fontsize=9); ax.grid()

    fig.tight_layout()
    for ext in ("png", "pdf"):
        path = os.path.join(output_dir, f"pruning_vs_vanilla.{ext}")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[plot] Saved → {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results", default="results/benchmark_results.json",
                   help="Path to JSON produced by benchmark.py")
    p.add_argument("--output-dir", default="results",
                   help="Directory for output figures")
    p.add_argument("--log-interval", type=int, default=10,
                   help="Log interval used during benchmarking (default: 10)")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse()

    if not os.path.isfile(args.results):
        print(f"[plot] ERROR: results file not found: {args.results}")
        print("       Run `python benchmark.py` first.")
        raise SystemExit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    data = load(args.results)

    make_overview_figure(data, args.output_dir, args.log_interval)
    make_active_seqs_figure(data, args.output_dir, args.log_interval)
    make_memory_figure(data, args.output_dir, args.log_interval)
    make_pruning_comparison_figure(data, args.output_dir, args.log_interval)

    print("[plot] All figures generated.")
