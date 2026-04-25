"""
Vanilla HuggingFace batch inference baseline.

All N prompts are padded to the same length and processed together until the
longest sequence reaches max_new_tokens – identical to the default
`model.generate()` behaviour.  Sequences that finish early continue to occupy
full GPU compute/memory because HuggingFace's generate() does not drop them.

A LogitsProcessor hook is injected to record, at regular intervals:
  - current GPU memory allocation
  - how many sequences have *already* generated an EOS token (i.e. are
    logically done but still padded-along in the batch)
  - the computed batch size (always == original batch size for vanilla)
"""
from __future__ import annotations

import time
from typing import List, Optional, Set

import torch
from transformers import LogitsProcessor

from .metrics import (
    BenchmarkResult,
    get_gpu_memory_mb,
    get_peak_gpu_memory_mb,
    reset_peak_memory,
)

# Qwen3 uses two EOS token ids
QWEN3_EOS_IDS: List[int] = [151643, 151645]


# ---------------------------------------------------------------------------
# Step-level metrics hook
# ---------------------------------------------------------------------------

class _StepLogger(LogitsProcessor):
    """Records GPU memory and logical-finish counts at each generation step."""

    def __init__(
        self,
        batch_size: int,
        eos_token_ids: Set[int],
        log_interval: int = 10,
    ) -> None:
        self.batch_size = batch_size
        self.eos_ids = eos_token_ids
        self.log_interval = log_interval
        self.step = 0

        self.memory_log: List[float] = []
        self.computed_log: List[int] = []   # always batch_size for vanilla
        self.finished_log: List[int] = []   # sequences past their EOS token

        # Track per-sequence EOS hit using a bool tensor (filled lazily)
        self._finished: Optional[torch.Tensor] = None

    def __call__(
        self,
        input_ids: torch.Tensor,   # [B, seq_len_so_far]
        scores: torch.Tensor,
    ) -> torch.Tensor:
        B = input_ids.shape[0]

        if self._finished is None:
            self._finished = torch.zeros(B, dtype=torch.bool, device=input_ids.device)

        # Check the most recently appended token for EOS
        if input_ids.shape[1] > 0:
            last = input_ids[:, -1]
            for eid in self.eos_ids:
                self._finished |= last.eq(eid)

        if self.step > 0 and self.step % self.log_interval == 0:
            self.memory_log.append(get_gpu_memory_mb())
            self.computed_log.append(B)                       # still whole batch
            self.finished_log.append(int(self._finished.sum().item()))

        self.step += 1
        return scores


# ---------------------------------------------------------------------------
# VanillaInference
# ---------------------------------------------------------------------------

class VanillaInference:
    """Standard HuggingFace batch inference (no dynamic pruning)."""

    def __init__(self, model, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer

        # Build EOS set from tokenizer + Qwen3 defaults
        eos: Set[int] = set(QWEN3_EOS_IDS)
        if tokenizer.eos_token_id is not None:
            eos.add(int(tokenizer.eos_token_id))
        self.eos_ids = eos

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
        Tokenise `prompts`, run `model.generate()`, and return metrics.

        Generation follows Qwen3 defaults: do_sample=True, temperature=0.6,
        top_p=0.95, top_k=20.
        """
        device = next(self.model.parameters()).device
        batch_size = len(prompts)

        # Left-padding is required for batched causal-LM generation
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,
        ).to(device)

        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # Metrics logger hook
        logger = _StepLogger(
            batch_size=batch_size,
            eos_token_ids=self.eos_ids,
            log_interval=log_interval,
        )

        reset_peak_memory()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                pad_token_id=pad_id,
                eos_token_id=sorted(self.eos_ids),
                logits_processor=[logger],
            )

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mem = get_peak_gpu_memory_mb()

        # Count generated (non-pad) tokens
        input_len = inputs.input_ids.shape[1]
        new_ids = output_ids[:, input_len:]
        total_tokens = int((new_ids != pad_id).sum().item())

        return BenchmarkResult(
            method="vanilla",
            batch_size=batch_size,
            total_time_s=elapsed,
            peak_memory_mb=peak_mem,
            tokens_per_second=total_tokens / elapsed,
            total_tokens=total_tokens,
            memory_over_time=logger.memory_log,
            computed_seqs_over_time=logger.computed_log,
            finished_seqs_over_time=logger.finished_log,
        )
