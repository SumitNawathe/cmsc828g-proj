"""
Lightweight metrics helpers for GPU memory and timing benchmarks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------

def get_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 ** 2
    return 0.0


def get_peak_gpu_memory_mb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    method: str          # "vanilla" | "efficient"
    batch_size: int
    total_time_s: float
    peak_memory_mb: float
    tokens_per_second: float
    total_tokens: int

    # Per-step logs (recorded every `log_interval` steps)
    memory_over_time: List[float] = field(default_factory=list)
    # For vanilla: number of sequences the model *actually computes* over (stays flat).
    # For efficient: number of sequences still actively generating (decreases).
    computed_seqs_over_time: List[int] = field(default_factory=list)
    # For vanilla only: how many sequences have already hit EOS but are still padded.
    finished_seqs_over_time: List[int] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "batch_size": self.batch_size,
            "total_time_s": self.total_time_s,
            "peak_memory_mb": self.peak_memory_mb,
            "tokens_per_second": self.tokens_per_second,
            "total_tokens": self.total_tokens,
            "memory_over_time": self.memory_over_time,
            "computed_seqs_over_time": self.computed_seqs_over_time,
            "finished_seqs_over_time": self.finished_seqs_over_time,
        }
