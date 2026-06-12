"""
Model inference module.

Loads a HuggingFace causal-LM and generates text completions in batches.
"""
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class ModelGenerator:
    """
    Wraps a HuggingFace causal language model for batch text generation.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model ID or local path (e.g. "EleutherAI/pythia-70m").
    device : str
        "auto", "cuda", or "cpu".
    torch_dtype : str
        "float32", "float16", or "bfloat16".
    max_new_tokens : int
        Maximum tokens to generate per prompt.
    do_sample : bool
        If False, uses greedy decoding (deterministic). Recommended for experiments.
    batch_size : int
        Number of prompts to process in one forward pass.
    hf_token : str, optional
        HuggingFace API token for private models.
    """

    def __init__(
        self,
        model_name_or_path: str,
        device: str = 'auto',
        torch_dtype: str = 'bfloat16',
        max_new_tokens: int = 100,
        do_sample: bool = False,
        batch_size: int = 8,
        hf_token: Optional[str] = None,
    ):
        self.model_name_or_path = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.batch_size = batch_size

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        # Resolve dtype
        dtype_map = {
            'float32': torch.float32,
            'float16': torch.float16,
            'bfloat16': torch.bfloat16,
        }
        self.torch_dtype = dtype_map.get(torch_dtype, torch.bfloat16)

        # Resolve device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        logger.info(f"Loading tokenizer: {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            token=hf_token,
            padding_side='left',   # left-pad for causal LM batched inference
        )
        # Many causal LMs lack a pad token; reuse EOS
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        logger.info(f"Loading model: {model_name_or_path} on {self.device} ({torch_dtype})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=self.torch_dtype,
            device_map=device if device == 'auto' else None,
            token=hf_token,
        )
        if device != 'auto':
            self.model = self.model.to(self.device)
        self.model.eval()
        logger.info("Model loaded and in eval mode")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _clean(text: str) -> str:
        """Strip lone surrogate code points that tokenizers reject."""
        if not isinstance(text, str):
            text = str(text) if text is not None else ''
        return text.encode('utf-8', errors='ignore').decode('utf-8')

    def _generate_batch(self, texts: List[str]) -> List[str]:
        import torch

        # Strip surrogates — tokenizer raises TypeError on lone surrogates
        texts = [self._clean(t) for t in texts]

        try:
            inputs = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            input_len = inputs['input_ids'].shape[1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=self.do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only newly generated tokens (strip the prompt)
            generated_ids = outputs[:, input_len:]
            decoded = self.tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
            return decoded

        except Exception as exc:
            logger.warning(
                f"_generate_batch: batch of {len(texts)} failed "
                f"({type(exc).__name__}: {exc}). "
                f"Falling back to per-sample generation."
            )
            # Fall back: try each sample individually, return '' on failure
            results = []
            for text in texts:
                try:
                    enc = self.tokenizer(
                        text,
                        return_tensors='pt',
                        truncation=True,
                        max_length=512,
                    ).to(self.device)
                    input_len = enc['input_ids'].shape[1]
                    with torch.no_grad():
                        out = self.model.generate(
                            **enc,
                            max_new_tokens=self.max_new_tokens,
                            do_sample=self.do_sample,
                            pad_token_id=self.tokenizer.pad_token_id,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                    results.append(
                        self.tokenizer.decode(
                            out[0, input_len:], skip_special_tokens=True
                        )
                    )
                except Exception as inner_exc:
                    logger.warning(
                        f"  Sample skipped — {type(inner_exc).__name__}: {inner_exc}"
                    )
                    results.append('')
            return results

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def generate(self, texts: List[str]) -> List[str]:
        """
        Generate completions for a list of prompt texts.

        Parameters
        ----------
        texts : List[str]
            Input prompts.

        Returns
        -------
        List[str]
            Generated continuations, one per prompt (prompt text NOT included).
        """
        results: List[str] = []
        total = len(texts)
        for start in range(0, total, self.batch_size):
            batch = texts[start: start + self.batch_size]
            logger.debug(f"Generating batch {start // self.batch_size + 1} "
                         f"({len(batch)} samples)")
            completions = self._generate_batch(batch)
            results.extend(completions)
        return results
