"""
CoDeC — Contamination Detection via Context.
Re-implemented from Related_work/Codec/contamination_detection_demo.ipynb
adapted to the src_v2 architecture.

Method: compare model log-probability of a target sample with and without
in-context examples from the same dataset. If confidence is *higher* without
context the model has likely already seen the distribution (contaminated).

Reference: https://www.arxiv.org/abs/2510.27055
"""
from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from safetensors.torch import load_file as load_safetensors
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from baseline.checkpoint_paths import resolve_hf_checkpoint

logger = logging.getLogger(__name__)


class ModelHandler:
    """Load a causal LM and expose token-level log-probability access."""

    def __init__(
        self,
        model_name: Optional[str],
        device: str = "auto",
        checkpoint_path: Optional[str] = None,
        verbose: bool = False,
        repo_root: Optional[Union[str, Path]] = None,
    ):
        raw = checkpoint_path or model_name
        if raw is None:
            raise ValueError("model_name or checkpoint_path is required")

        root = Path(repo_root).resolve() if repo_root is not None else None
        model_dir, tokenizer_dir = resolve_hf_checkpoint(raw, repo_root=root)
        self.model_name = str(model_dir)

        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if verbose:
            print(f"Loading model {model_dir} on {self.device}…")
            if model_dir != tokenizer_dir:
                print(f"  tokenizer/config from {tokenizer_dir}")

        dtype = torch.float16 if self.device == "cuda" else torch.float32
        local_kw = {"local_files_only": True}
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), **local_kw)
        config = AutoConfig.from_pretrained(str(tokenizer_dir), **local_kw)
        self.model = AutoModelForCausalLM.from_pretrained(
            str(tokenizer_dir),
            config=config,
            torch_dtype=dtype,
            **local_kw,
        )
        if model_dir != tokenizer_dir:
            weights_file = model_dir / "model.safetensors"
            if weights_file.is_file():
                state = load_safetensors(str(weights_file))
                self.model.load_state_dict(state, strict=False)
            else:
                logger.warning("Epoch weights expected in %s but model.safetensors missing", model_dir)
        self.model = self.model.to(self.device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if verbose:
            print("Model loaded.")

    def get_logprobs(self, text: str) -> np.ndarray:
        """Return per-token log-probabilities (length = n_tokens - 1)."""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            log_probs = torch.log_softmax(outputs.logits, dim=-1)
            input_ids = inputs["input_ids"][0]
            token_lp = [
                log_probs[0, i, input_ids[i + 1]].item()
                for i in range(len(input_ids) - 1)
            ]
        return np.array(token_lp)


class CoDeC:
    """Core CoDeC contamination detector."""

    def __init__(self, token_range: Tuple[int, int] = (10, -1)):
        self.token_range = token_range

    def detect(
        self,
        target_text: str,
        context_examples: List[str],
        model_handler: ModelHandler,
    ) -> Optional[float]:
        """
        Returns 1.0 (contaminated) or 0.0 (not contaminated), or None if the
        target text is too short to score reliably.
        """
        lp_no_ctx = model_handler.get_logprobs(target_text)

        context = "\n\n".join(context_examples)
        text_with_ctx = (context + "\n\n" + target_text) if context_examples else target_text
        lp_with_ctx = model_handler.get_logprobs(text_with_ctx)

        n = len(lp_no_ctx)
        lp_target_from_ctx = lp_with_ctx[-n:]

        start, end = self.token_range
        end = min(end if end >= 0 else n, n)
        if start >= n - 1:
            return None

        conf_no_ctx = float(np.mean(lp_no_ctx[start:end]))
        conf_with_ctx = float(np.mean(lp_target_from_ctx[start:end]))
        return 1.0 if conf_no_ctx > conf_with_ctx else 0.0

    # Convenience wrapper that matches the pipeline API
    detect_contamination = detect


def contamination_detection_pipeline(
    model: str | ModelHandler,
    dataset: List[str],
    num_context_examples: int = 1,
    max_dataset_size: int = 1000,
    max_sample_length: int = 2048,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run CoDeC on a dataset and return aggregate results."""
    random.seed(seed)
    np.random.seed(seed)

    model_handler = model if isinstance(model, ModelHandler) else ModelHandler(model)
    detector = CoDeC()

    # Trim long samples
    if max_sample_length:
        dataset = [s[:max_sample_length] for s in dataset]

    if max_dataset_size and len(dataset) > max_dataset_size:
        dataset = random.sample(dataset, max_dataset_size)

    sample_scores: List[float] = []
    for i, target in enumerate(dataset):
        pool = dataset[:i] + dataset[i + 1 :]
        ctx = random.sample(pool, min(num_context_examples, len(pool)))
        score = detector.detect(target, ctx, model_handler)
        if score is not None:
            sample_scores.append(score)

    overall = float(np.mean(sample_scores)) if sample_scores else 0.0
    return {
        "model_name": model_handler.model_name,
        "overall_contamination_score": overall,
        "sample_scores": sample_scores,
        "num_samples": len(dataset),
        "num_contaminated": int(sum(sample_scores)),
    }
