"""
strategies.py — Batching strategies for multi-GPU scheduled inference.

Ported from simulation_writeup_2.ipynb with modifications for real inference.
"""

import copy
import numpy as np
import scipy.stats
from abc import ABC, abstractmethod


class BatchingStrategy(ABC):
    """
    Base class for batching strategies.

    A strategy produces an ordering of batches for each epoch.
    Each batch is a list of sample indices (keys). Each key must appear
    exactly once across all batches in an epoch. For each key, G rollouts
    are generated (all G rollouts for a given key are in the same batch).

    After each batch completes, `ingest_run_data` is called with the
    observed rollout lengths, allowing adaptive strategies to update
    their models.
    """

    def __init__(self, keys: list[int], G: int):
        self.keys = list(keys)
        self.G = G

    @abstractmethod
    def ingest_run_data(self, key_length_data: list[tuple[int, int]]):
        """
        Receive observed (key, rollout_length) pairs from a completed batch.
        """
        pass

    @abstractmethod
    def get_ordering(self) -> list[list[int]]:
        """
        Return the batch ordering for the next epoch.
        Each element is a list of sample keys to include in that batch.
        """
        pass


class BaselineStrategy(BatchingStrategy):
    """
    Baseline: each prompt is in its own batch with G rollouts.
    No adaptation, no multi-prompt batching.
    """

    def __init__(self, keys: list[int], G: int, bg: int = 1):
        super().__init__(keys, G)
        self.bg = bg

    def ingest_run_data(self, key_length_data: list[tuple[int, int]]):
        pass  # No adaptation needed

    def get_ordering(self) -> list[list[int]]:
        ans = []
        b = []
        for x in self.keys:
            b.append(x)
            if len(b) == self.bg:
                ans.append(b)
                b = []
        if b:
            ans.append(b)
        return ans


class LogTSumStrategy(BatchingStrategy):
    """
    Adaptive strategy using Log-t distribution modeling.

    On epoch 0 (no data), falls back to BaselineStrategy ordering.
    On subsequent epochs, fits a t-distribution to log(rollout_lengths)
    for each prompt, then greedily groups prompts into batches while
    keeping the Monte Carlo estimate of P[sum_lengths > mem_limit] < alpha.
    """

    def __init__(
        self,
        keys: list[int],
        G: int,
        mem_limit: int = 250000,
        alpha: float = 0.01,
        mc_nsamples: int = 10_000,
        init_seed: int = 42,
    ):
        super().__init__(keys, G)
        self.mem_limit = mem_limit
        self.alpha = alpha
        self.mc_nsamples = mc_nsamples
        self.rng = np.random.default_rng(init_seed)

        # Collected rollout lengths per key
        self.times: dict[int, list[int]] = {key: [] for key in keys}
        # Fitted distribution parameters per key: (df, loc, scale)
        self.dist_params: dict[int, tuple[float, float, float]] = {
            key: (0.0, 0.0, 0.0) for key in keys
        }

    def ingest_run_data(self, key_length_data: list[tuple[int, int]]):
        """Update collected lengths and refit distributions for affected keys."""
        affected_keys = set()
        for (k, length) in key_length_data:
            self.times[k].append(length)
            affected_keys.add(k)

        for k in affected_keys:
            if len(self.times[k]) >= 3:
                # Need at least 3 data points to fit a t-distribution
                try:
                    self.dist_params[k] = scipy.stats.t.fit(np.log(self.times[k]))
                except Exception:
                    # If fitting fails, keep old params
                    pass

    def _has_data(self) -> bool:
        """Check if we have enough data to use the adaptive strategy."""
        return any(len(self.times[k]) >= 3 for k in self.keys)

    def simulate_lengths(self) -> dict[int, np.ndarray]:
        """
        For each key, simulate G rollout lengths using the fitted Log-t
        distribution, and return the sum across G rollouts (total tokens
        that key would consume in a batch).
        """
        output = {}
        for k, (df, loc, scale) in self.dist_params.items():
            if df <= 0 or scale <= 0:
                # No valid fit — use empirical mean as a constant
                if self.times[k]:
                    mean_val = np.mean(self.times[k])
                    output[k] = np.full(self.mc_nsamples, mean_val * self.G)
                else:
                    output[k] = np.zeros(self.mc_nsamples)
                continue

            t_samples = self.rng.standard_t(max(df, 1.01), size=(self.G, self.mc_nsamples))
            log_samples = np.clip(loc + scale * t_samples, a_min=None, a_max=20.0)
            length_samples = np.exp(log_samples)
            output[k] = length_samples.sum(axis=0)  # sum over G rollouts

        return output

    def get_ordering(self) -> list[list[int]]:
        if not self._has_data():
            # Epoch 0 fallback: one prompt per batch
            return [[k] for k in self.keys]

        # Sort by mean + 0.5*std of observed lengths (ascending = easiest first)
        means_and_keys = []
        for k in self.keys:
            if self.times[k]:
                m = np.mean(self.times[k])
                s = np.std(self.times[k])
                means_and_keys.append((m + 0.5 * s, k))
            else:
                means_and_keys.append((0.0, k))
        means_and_keys.sort()

        P = [t[1] for t in means_and_keys]  # sorted key list
        simulated_lengths = self.simulate_lengths()

        B: list[list[int]] = []
        while P:
            p = P.pop(0)
            b = [p]
            while P:
                candidate = P[0]
                b_new = b + [candidate]
                # Sum simulated lengths for the candidate batch
                b_new_simul: np.ndarray = sum(simulated_lengths[key] for key in b_new)
                # Check if OOM probability is below threshold
                if (b_new_simul >= self.mem_limit).mean() < self.alpha:
                    b = b_new
                    P.pop(0)
                else:
                    break
            B.append(b)

        # Return hardest (largest) batches first, matching notebook
        return list(reversed(B))


class LogTMaxStrategy(BatchingStrategy):
    """
    Adaptive strategy using Log-t distribution modeling.

    On epoch 0 (no data), falls back to BaselineStrategy ordering.
    On subsequent epochs, fits a t-distribution to log(rollout_lengths)
    for each prompt, then greedily groups prompts into batches while
    keeping the Monte Carlo estimate of P[max_lengths * B > mem_limit] < alpha.
    """

    def __init__(
        self,
        keys: list[int],
        G: int,
        mem_limit: int = 250000,
        alpha: float = 0.001,
    ):
        super().__init__(keys, G)
        self.mem_limit = mem_limit
        self.alpha = alpha

        # Collected rollout lengths per key
        self.times: dict[int, list[int]] = {key: [] for key in keys}
        # Fitted distribution parameters per key: (df, loc, scale)
        self.dist_params: dict[int, tuple[float, float, float]] = {
            key: (0.0, 0.0, 0.0) for key in keys
        }

    def ingest_run_data(self, key_length_data: list[tuple[int, int]]):
        """Update collected lengths and refit distributions for affected keys."""
        affected_keys = set()
        for (k, length) in key_length_data:
            self.times[k].append(length)
            affected_keys.add(k)

        for k in affected_keys:
            if len(self.times[k]) >= 3:
                # Need at least 3 data points to fit a t-distribution
                try:
                    self.dist_params[k] = scipy.stats.t.fit(np.log(self.times[k]))
                except Exception:
                    # If fitting fails, keep old params
                    pass

    def _has_data(self) -> bool:
        """Check if we have enough data to use the adaptive strategy."""
        return any(len(self.times[k]) >= 3 for k in self.keys)

    def prob_exceed_limit(self, keys: list[int], limit: int) -> float:
        probs = []
        for k in keys:
            df, loc, scale = self.dist_params[k]
            if df <= 0 or scale <= 0:
                if self.times[k] and np.mean(self.times[k]) > limit:
                    probs.append(1.0)
                else:
                    probs.append(0.0)
                continue
            prob = 1 - scipy.stats.t.cdf(np.log(limit), df, loc, scale)
            probs.append(prob)
        return 1 - np.array([1 - p for p in probs]).prod()

    def get_ordering(self) -> list[list[int]]:
        if not self._has_data():
            # Epoch 0 fallback: one prompt per batch
            return [[k] for k in self.keys]

        # Sort by mean + 0.5*std of observed lengths (ascending = easiest first)
        means_and_keys = []
        for k in self.keys:
            if self.times[k]:
                m = np.mean(self.times[k])
                s = np.std(self.times[k])
                means_and_keys.append((m + 0.5 * s, k))
            else:
                means_and_keys.append((0.0, k))
        means_and_keys.sort()

        P = [t[1] for t in means_and_keys]  # sorted key list

        B: list[list[int]] = []
        while P:
            p = P.pop(0)
            b = [p]
            while P:
                candidate = P[0]
                b_new = b + [candidate]
                limit = self.mem_limit // (self.G * len(b_new))
                # Check if OOM probability is below threshold
                if self.prob_exceed_limit(b_new * self.G, limit) < self.alpha:
                    b = b_new
                    P.pop(0)
                else:
                    break
            B.append(b)

        # Return hardest (largest) batches first, matching notebook
        return list(reversed(B))
