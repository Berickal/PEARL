"""
ACR — Adversarial Compression Ratio detector.
Re-implemented from Related_work/acr-memorization-main/ adapted for src_v2.

Method: a string is considered memorized if it can be elicited by a prompt
*shorter than itself* found via gradient-based (GCG) or random discrete
optimization. Requires gradient access → white-box.

This module wraps the original prompt_optimization code and exposes a clean
interface consistent with the other baselines.

Reference: Schwarzschild et al., "Rethinking LLM Memorization through the
Lens of Adversarial Compression", arXiv:2404.15146.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch

# Make original ACR code importable
_ACR_ROOT = Path(__file__).parent.parent.parent / "Related_work" / "acr-memorization-main"
if str(_ACR_ROOT) not in sys.path:
    sys.path.insert(0, str(_ACR_ROOT))


@dataclass
class ACRConfig:
    optimizer: str = "gcg"          # "gcg" or "random_search"
    num_free_tokens: int = 10       # maximum adversarial prefix length
    num_steps: int = 200
    batch_size: int = 100
    topk: int = 256
    lr: float = 0.01
    device: str = "auto"


class ACRDetector:
    """
    White-box memorization detector based on adversarial compression.

    Parameters
    ----------
    model : transformers AutoModelForCausalLM
    tokenizer : transformers AutoTokenizer
    config : ACRConfig
    """

    def __init__(self, model, tokenizer, config: Optional[ACRConfig] = None):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or ACRConfig()

        if self.config.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = self.config.device

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def compression_ratio(self, target_str: str) -> float:
        """
        Find the shortest adversarial prompt that elicits `target_str`.
        Returns len(optimal_prompt) / len(target_str) in *token* space.

        A ratio < 1 means the string can be compressed → memorized.
        """
        import prompt_optimization as prompt_opt

        input_ids, free_token_slice, input_slice, target_slice, loss_slice = (
            prompt_opt.prep_text(
                " ",          # empty input string
                target_str,
                self.tokenizer,
                "",           # no system prompt
                ("", ""),     # no chat template
                self.config.num_free_tokens,
                self.device,
            )
        )

        if self.config.optimizer == "gcg":
            solution = prompt_opt.optimize_gcg(
                self.model,
                input_ids,
                input_slice,
                free_token_slice,
                target_slice,
                loss_slice,
                self.config.num_steps,
                batch_size=self.config.batch_size,
                topk=self.config.topk,
            )
        elif self.config.optimizer == "random_search":
            solution = prompt_opt.optimize_random_search(
                self.model,
                input_ids,
                input_slice,
                free_token_slice,
                target_slice,
                loss_slice,
                self.config.num_steps,
                batch_size=self.config.batch_size,
            )
        else:
            raise ValueError(f"Unknown optimizer '{self.config.optimizer}'")

        optimized_ids = solution["input_ids"]
        prompt_len = len(optimized_ids[input_slice])
        target_len = len(input_ids[target_slice])
        return prompt_len / max(target_len, 1)

    def is_memorized(self, target_str: str, threshold: float = 1.0) -> Tuple[float, bool]:
        """
        Returns (compression_ratio, is_memorized).
        is_memorized is True when compression_ratio < threshold.
        """
        ratio = self.compression_ratio(target_str)
        return ratio, ratio < threshold

    def score_dataset(
        self, targets: List[str], threshold: float = 1.0
    ) -> List[Tuple[float, bool]]:
        return [self.is_memorized(t, threshold) for t in targets]
