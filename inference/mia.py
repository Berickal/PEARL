"""
Membership Inference Attack (MIA) scoring methods.

Three methods are implemented:

  Loss               — Average per-token cross-entropy loss on the input text.
                       Lower loss → model assigns high probability to the text
                       → evidence of memorization.

  Min-K% Prob        — Average of the minimum-k% token log-probabilities
                       (Shi et al., 2024 "Detecting Pretraining Data from LLMs").
                       Higher (less negative) → token distribution is more peaked
                       → evidence of memorization.

  Neighborhood Attack — score = mean_loss(neighbors) − loss(original).
                        Positive → original has lower loss than its perturbed
                        neighbors → evidence of memorization.
                        Neighbors should be drawn from the [0.90-1.00] similarity
                        range (very close perturbations of the original text).

Convention: for all three methods, a *higher* returned score means stronger
evidence of memorization, so callers can compare them on a common scale.
Concretely:
    mia_loss        → negate to get a "memorized-if-high" score (lower raw loss = memorized)
    mia_min_k       → already "higher = memorized" (less-negative mean log-prob)
    mia_neighborhood → already "higher = memorized" (positive = original easier than neighbors)
"""
import logging
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MIAScorer:
    """
    Computes MIA scores for a text using an already-loaded causal LM.

    Designed to share the model instance with ModelGenerator — no extra
    model loading cost.

    Parameters
    ----------
    model :
        Loaded HuggingFace AutoModelForCausalLM (already on the right device,
        in eval mode).
    tokenizer :
        Matching AutoTokenizer.
    device : str
        Device string ("cpu", "cuda", etc.).
    max_length : int
        Maximum token length; inputs are truncated to this length.
    min_k : float
        Fraction of tokens used in the Min-K% attack (default 0.2 = 20%).
    """

    def __init__(
        self,
        model,
        tokenizer,
        device: str = 'cpu',
        max_length: int = 512,
        min_k: float = 0.2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length
        self.min_k = min_k

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _token_log_probs(self, text: str) -> Optional[torch.Tensor]:
        """
        Return a 1-D float tensor of per-token log-probabilities for *text*.

        The tensor has length T-1 (the first token has no antecedent prediction).
        Returns None when the text tokenises to fewer than 2 tokens.
        """
        if text is None:
            return None
        if not isinstance(text, str):
            text = str(text)
        # Strip lone surrogates that the tokenizer's UTF-8 encoder rejects
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        if not text.strip():
            return None

        try:
            enc = self.tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=self.max_length,
                padding=False,
            ).to(self.device)

            input_ids = enc['input_ids']   # [1, T]
            if input_ids.shape[1] < 2:
                logger.debug("Text tokenises to fewer than 2 tokens, skipping MIA.")
                return None

            with torch.no_grad():
                logits = self.model(**enc).logits   # [1, T, V]

            # logits[t] predicts token[t+1]  →  shift by one
            shift_logits = logits[0, :-1, :]      # [T-1, V]
            shift_labels = input_ids[0, 1:]        # [T-1]

            log_probs = F.log_softmax(shift_logits, dim=-1)
            token_log_probs = log_probs[
                torch.arange(shift_labels.shape[0], device=self.device),
                shift_labels,
            ]
            return token_log_probs   # [T-1]

        except Exception as exc:
            logger.warning(
                f"MIA._token_log_probs: skipping sample — {type(exc).__name__}: {exc}"
            )
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_loss(self, text: str) -> Optional[float]:
        """
        Average per-token cross-entropy loss.

        Returns None if the text is too short to score.
        Interpretation: lower raw value → more memorized.
        """
        token_lp = self._token_log_probs(text)
        if token_lp is None:
            return None
        loss = -token_lp.mean().item()
        return round(loss, 6)

    def compute_min_k(self, text: str, k: Optional[float] = None) -> Optional[float]:
        """
        Min-K% Prob score (Shi et al., 2024).

        Computes the average of the *k%* lowest token log-probabilities.
        Higher (less negative) → evidence of memorization.

        Returns None if the text is too short.
        """
        k = k if k is not None else self.min_k
        token_lp = self._token_log_probs(text)
        if token_lp is None:
            return None

        n = len(token_lp)
        k_count = max(1, int(n * k))
        min_k_vals, _ = torch.topk(token_lp, k_count, largest=False)
        score = min_k_vals.mean().item()
        return round(score, 6)

    def compute_neighborhood_score(
        self,
        text: str,
        neighbors: List[str],
    ) -> Optional[float]:
        """
        Neighborhood Attack score.

        score = mean_loss(neighbors) − loss(original)

        Positive → original has lower loss than its perturbated neighbors
        → evidence of memorization.

        Returns None if there are no scorable neighbors or if the original
        text is too short.
        """
        if not neighbors:
            return None

        original_loss = self.compute_loss(text)
        if original_loss is None:
            return None

        neighbor_losses = []
        for nbr in neighbors:
            l = self.compute_loss(nbr)
            if l is not None:
                neighbor_losses.append(l)

        if not neighbor_losses:
            return None

        mean_neighbor_loss = sum(neighbor_losses) / len(neighbor_losses)
        score = mean_neighbor_loss - original_loss
        return round(score, 6)

    def score_all(
        self,
        text: str,
        neighbors: Optional[List[str]] = None,
    ) -> Dict[str, Optional[float]]:
        """
        Compute all three MIA scores in a single call.

        Returns a dict with keys:
            mia_loss            (lower raw value = more memorized)
            mia_min_k           (higher value = more memorized)
            mia_neighborhood    (higher value = more memorized)
        """
        return {
            'mia_loss': self.compute_loss(text),
            'mia_min_k': self.compute_min_k(text),
            'mia_neighborhood': self.compute_neighborhood_score(
                text, neighbors or []
            ),
        }
