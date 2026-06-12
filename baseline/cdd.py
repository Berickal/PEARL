"""
CDD — Contamination Detection via Distribution peakedness.
Re-implemented from Related_work/CDD-TED4LLMs-main/CDD.py adapted for src_v2.

Method: generate N sampled completions and 1 greedy completion from the model
for each instance. Compute the edit-distance distribution between sampled and
greedy completions. A "peaked" distribution (most samples very close to greedy)
indicates the model is highly confident → likely contaminated / memorized.

Reference: CDD paper (see Related_work/CDD-TED4LLMs-main/README.md)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np


@dataclass
class CDDConfig:
    alpha: float = 0.05   # distance threshold as fraction of greedy-output length
    xi: float = 0.01      # peakedness threshold (ratio of samples within α*len)
    n_samples: int = 20   # number of stochastic completions per instance
    max_token_length: int = 100


class CDDDetector:
    """
    Black-box CDD detector.

    Parameters
    ----------
    generate_fn : callable(prompt, temperature, n) -> List[str]
        Generate `n` completions for `prompt` at the given temperature.
    tokenize_fn : callable(text) -> List[int]
        Tokenize text to a list of integer token IDs.
    config : CDDConfig
    """

    def __init__(
        self,
        generate_fn: Callable[[str, float, int], List[str]],
        tokenize_fn: Callable[[str], List[int]],
        config: Optional[CDDConfig] = None,
    ):
        self.generate = generate_fn
        self.tokenize = tokenize_fn
        self.config = config or CDDConfig()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _levenshtein(a: List[int], b: List[int]) -> int:
        if len(a) > len(b):
            a, b = b, a
        dists = list(range(len(a) + 1))
        for j, cb in enumerate(b):
            new_dists = [j + 1]
            for i, ca in enumerate(a):
                if ca == cb:
                    new_dists.append(dists[i])
                else:
                    new_dists.append(1 + min(dists[i], dists[i + 1], new_dists[-1]))
            dists = new_dists
        return dists[-1]

    def _edit_distances(
        self,
        samples: List[str],
        greedy: str,
    ) -> Tuple[List[int], int]:
        g_tokens = self.tokenize(greedy)[: self.config.max_token_length]
        max_len = len(g_tokens)
        distances = []
        for s in samples:
            s_tokens = self.tokenize(s)[: len(g_tokens)]
            distances.append(self._levenshtein(g_tokens, s_tokens))
            max_len = max(max_len, len(s_tokens))
        return distances, max_len

    def _peakedness(self, distances: List[int], max_len: int) -> float:
        threshold = self.config.alpha * max_len
        return sum(1 for d in distances if d <= threshold) / len(distances)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def score_instance(self, prompt: str) -> Tuple[float, bool]:
        """
        Compute the CDD peakedness score for a single instance.

        Returns (peak_score, is_memorized).
        """
        # Greedy completion (temperature=0, 1 sample)
        greedy_completions = self.generate(prompt, temperature=0.0, n=1)
        greedy = greedy_completions[0] if greedy_completions else ""

        # Stochastic completions
        stochastic = self.generate(prompt, temperature=0.8, n=self.config.n_samples)

        distances, max_len = self._edit_distances(stochastic, greedy)
        peak = self._peakedness(distances, max_len)
        is_memorized = peak > self.config.xi
        return peak, is_memorized

    def score_dataset(self, prompts: List[str]) -> List[Tuple[float, bool]]:
        return [self.score_instance(p) for p in prompts]

    def score_from_completions(
        self,
        greedy: str,
        stochastic_samples: List[str],
    ) -> Tuple[float, bool]:
        """Compute peakedness from a greedy completion and stochastic samples (no model I/O).

        Use this when completions are produced elsewhere (e.g. batched HF generation).
        """
        if not stochastic_samples:
            return 0.0, False
        distances, max_len = self._edit_distances(stochastic_samples, greedy)
        peak = self._peakedness(distances, max_len)
        return peak, peak > self.config.xi
