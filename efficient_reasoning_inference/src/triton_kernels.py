"""
Custom Triton kernels for efficient token sampling during inference.

Provides:
  - triton_greedy_decode : two-pass row-wise argmax (avoids allocating a
    full sorted copy of the vocab as torch.argmax can do internally).
  - triton_topk_sample   : fused temperature scaling + top-k filtering +
    multinomial sample (falls back to PyTorch when Triton is unavailable).

Both functions expose a HAS_TRITON flag so callers can decide whether to use
the Triton path or a pure-PyTorch fallback.
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True

    # ------------------------------------------------------------------
    # Pass 1: find the maximum logit value in each row
    # ------------------------------------------------------------------

    @triton.jit
    def _row_max_kernel(
        logits_ptr,    # float32 [B, V]
        out_ptr,       # float32 [B]
        V,             # vocab size (runtime int)
        BLOCK_V: tl.constexpr,
    ):
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

    # ------------------------------------------------------------------
    # Pass 2: find the first index whose value equals the row maximum
    # ------------------------------------------------------------------

    @triton.jit
    def _row_argmax_kernel(
        logits_ptr,    # float32 [B, V]
        max_ptr,       # float32 [B]  (from pass 1)
        out_ptr,       # int32   [B]
        V,
        BLOCK_V: tl.constexpr,
    ):
        row = tl.program_id(0)
        base = row * V
        global_max = tl.load(max_ptr + row)

        # Use V as an out-of-range sentinel; we want the *minimum* valid index.
        best_idx = V

        for start in range(0, V, BLOCK_V):
            offs = start + tl.arange(0, BLOCK_V)
            mask = offs < V
            vals = tl.load(logits_ptr + base + offs, mask=mask, other=-float("inf"))
            is_max = (vals == global_max) & mask
            # Replace non-max positions with sentinel V, then take block min.
            cand = tl.where(is_max, offs, V)
            block_min = tl.min(cand, axis=0)
            best_idx = tl.where(block_min < best_idx, block_min, best_idx)

        tl.store(out_ptr + row, best_idx)

    # ------------------------------------------------------------------
    # Public Python wrapper
    # ------------------------------------------------------------------

    def triton_greedy_decode(logits: torch.Tensor) -> torch.Tensor:
        """
        Row-wise argmax using two Triton kernel passes.

        Args:
            logits: float32 tensor of shape [B, V].  Must be contiguous.

        Returns:
            int64 tensor of shape [B] with the argmax index per row.
        """
        if not logits.is_contiguous():
            logits = logits.contiguous()

        B, V = logits.shape
        BLOCK_V = 1024  # tunable; 1024 fits comfortably in registers

        row_maxes = torch.empty(B, dtype=torch.float32, device=logits.device)
        argmax_out = torch.empty(B, dtype=torch.int32, device=logits.device)

        grid = (B,)
        _row_max_kernel[grid](logits, row_maxes, V, BLOCK_V=BLOCK_V)
        _row_argmax_kernel[grid](logits, row_maxes, argmax_out, V, BLOCK_V=BLOCK_V)

        return argmax_out.to(torch.int64)

    # ------------------------------------------------------------------
    # Fused temperature + top-k + sample  (Triton for greedy, PT for sampling)
    # ------------------------------------------------------------------

    def triton_topk_sample(
        logits: torch.Tensor,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        Sample next tokens.  Uses the Triton argmax kernel for greedy mode;
        falls back to PyTorch for stochastic sampling (top-k + top-p).
        """
        if greedy or temperature == 0.0:
            return triton_greedy_decode(logits)

        # Temperature scaling
        scaled = logits / temperature if temperature != 1.0 else logits

        # Top-k filter
        if top_k > 0:
            k = min(top_k, scaled.size(-1))
            topk_vals = torch.topk(scaled, k, dim=-1).values
            threshold = topk_vals[:, -1].unsqueeze(-1)
            scaled = scaled.masked_fill(scaled < threshold, float("-inf"))

        # Top-p (nucleus) filter
        if top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(scaled, descending=True, dim=-1)
            cum_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            # Shift right so we include the token that pushes cumprob over threshold
            remove = (cum_probs - torch.softmax(sorted_logits, dim=-1)) > top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            scaled = scaled.scatter(1, sorted_idx, sorted_logits)

        probs = torch.softmax(scaled, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)


except ImportError:  # Triton not installed – provide pure-PyTorch fallbacks
    HAS_TRITON = False

    def triton_greedy_decode(logits: torch.Tensor) -> torch.Tensor:  # type: ignore[misc]
        return torch.argmax(logits, dim=-1)

    def triton_topk_sample(  # type: ignore[misc]
        logits: torch.Tensor,
        temperature: float = 0.6,
        top_p: float = 0.95,
        top_k: int = 20,
        greedy: bool = False,
    ) -> torch.Tensor:
        if greedy or temperature == 0.0:
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
