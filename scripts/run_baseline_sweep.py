"""
Baseline memorization detection sweep — CDD, CoDeC, ACR.

Runs the three baselines from baseline/ on members and non-members across
fine-tuned checkpoints. Unlike the IPA/PEARL pipeline, no perturbations are
needed — each baseline operates directly on the original input texts.

Batch processing
----------------
CDD  : prompts are processed in batches of `--batch-size`.
       - 1 batched greedy generation pass
       - `--cdd-n-samples` batched stochastic passes (one pass per sample)
       Speedup ≈ batch_size× vs sequential.

CoDeC: both the "no-context" and "with-context" forward passes are batched
       across all samples using right-padded inputs + attention-mask slicing.
       Speedup ≈ batch_size× vs sequential.

ACR  : inherently sequential (iterative GCG optimisation per sample).

Usage (from src_v3/):
    python scripts/run_baseline_sweep.py \\
        --model checkpoints/EleutherAI_pythia-70m_fine_tuned/checkpoint-epoch-4 \\
        --epoch 4 --split both --methods cdd codec --batch-size 16

    # ACR is very slow (GCG optimisation per sample); avoid for large sweeps
    python scripts/run_baseline_sweep.py --model ... --epoch 4 --methods acr

    # Quick test: first 50 records per JSON file
    python scripts/run_baseline_sweep.py --model ... --epoch 4 --max-samples 50
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("baseline_sweep")

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline.cdd import CDDDetector, CDDConfig
from baseline.codec import CoDeC, ModelHandler
from utils import load_config

_DEFAULT_DATA_MEMBERS = "data/mimir_ngram_13_0.8_members.json"
_DEFAULT_DATA_NON_MEMBERS = "data/mimir_ngram_13_0.8_non_members.json"


# ── shared helpers ────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Strip lone surrogate code points that tokenisers reject."""
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def load_data(path: str, max_samples: Optional[int] = None) -> List[str]:
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    texts = []
    for r in records:
        text = r.get("input") or r.get("text") or "" if isinstance(r, dict) else str(r)
        texts.append(_clean(text))
    n_total = len(texts)
    if max_samples is not None:
        if max_samples <= 0:
            raise ValueError(f"max_samples must be positive, got {max_samples}")
        texts = texts[:max_samples]
        logger.info(
            "  using %d / %d samples from %s",
            len(texts), n_total, path,
        )
    return texts


def save_csv(rows: list, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    logger.info(f"Saved {len(rows)} rows → {path}")


def merge_results(method_rows: dict, n: int) -> List[dict]:
    df = pd.DataFrame([{"sample_idx": i} for i in range(n)])
    for rows in method_rows.values():
        if rows:
            df = df.merge(pd.DataFrame(rows), on="sample_idx", how="left")
    return df.to_dict(orient="records")


# ── low-level batch primitives ────────────────────────────────────────────────

def _batch_generate(
    prompts: List[str],
    model,
    tokenizer,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
    temperature: float = 1.0,
) -> List[str]:
    """
    Generate one completion per prompt in a single batched call.

    Uses LEFT padding so that all prompt endings are aligned at the same
    position — the generated tokens then start at index `padded_len` for
    every sequence in the batch, making slicing trivial.
    """
    if not prompts:
        return []

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        padded_len = enc["input_ids"].shape[1]
        gen_kwargs: dict = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.no_grad():
            out = model.generate(**enc, **gen_kwargs)

        # out[:, :padded_len] is the (padded) prompt; [padded_len:] is new tokens
        return [
            tokenizer.decode(out[i, padded_len:], skip_special_tokens=True)
            for i in range(len(prompts))
        ]

    except Exception as exc:
        logger.warning(f"_batch_generate failed ({exc}); falling back to per-sample.")
        results = []
        for p in prompts:
            try:
                enc = tokenizer(p, return_tensors="pt", truncation=True,
                                max_length=512).to(device)
                input_len = enc["input_ids"].shape[1]
                kw = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                          pad_token_id=tokenizer.pad_token_id)
                if do_sample:
                    kw["temperature"] = temperature
                with torch.no_grad():
                    out = model.generate(**enc, **kw)
                results.append(tokenizer.decode(out[0, input_len:], skip_special_tokens=True))
            except Exception as inner:
                logger.warning(f"  sample skipped: {inner}")
                results.append("")
        return results

    finally:
        tokenizer.padding_side = orig_side


def batched_logprobs(
    model,
    tokenizer,
    texts: List[str],
    device: str,
    batch_size: int,
) -> List[np.ndarray]:
    """
    Compute per-token log-probs for each text using RIGHT-padded batched
    forward passes.

    Returns a list of 1-D numpy arrays where array[j] = log P(token[j+1] |
    token[0..j]).  Each array has length = n_actual_tokens − 1.

    Right padding is used so that the actual tokens sit at the LEFT of each
    row; we recover the true length per sample from the attention mask and
    slice accordingly, ignoring the padding positions.
    """
    all_lp: List[np.ndarray] = []

    orig_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    try:
        for start in range(0, len(texts), batch_size):
            batch = [_clean(t) for t in texts[start:start + batch_size]]

            enc = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(device)

            input_ids      = enc["input_ids"]       # [B, L]
            attention_mask = enc["attention_mask"]  # [B, L]

            with torch.no_grad():
                logits = model(**enc).logits        # [B, L, V]

            log_probs = torch.log_softmax(logits, dim=-1)  # [B, L, V]

            for i in range(len(batch)):
                actual_len = int(attention_mask[i].sum().item())
                if actual_len < 2:
                    all_lp.append(np.array([], dtype=np.float32))
                    continue

                ids_i = input_ids[i, :actual_len]        # [T]
                lp_i  = log_probs[i, :actual_len - 1]    # [T-1, V]

                # gather log P(ids_i[t+1]) for t = 0..T-2
                token_lp = lp_i[
                    torch.arange(actual_len - 1, device=device), ids_i[1:]
                ].cpu().numpy()

                all_lp.append(token_lp)

            if (start // batch_size + 1) % 10 == 0:
                logger.info(f"    logprobs: {start + len(batch)}/{len(texts)}")

    finally:
        tokenizer.padding_side = orig_side

    return all_lp


# ── CDD (batched) ─────────────────────────────────────────────────────────────

def run_cdd(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    prompt_chars: int,
    n_samples: int,
    max_new_tokens: int,
    batch_size: int,
) -> List[dict]:
    """
    Batched CDD.

    Strategy:
      1. Extract prompt prefix (first `prompt_chars` chars) for every sample.
      2. One batched greedy pass  → greedy_completions[i].
      3. `n_samples` batched stochastic passes → stochastic[i] grows to a list
         of n_samples strings.
      4. Score peakedness of the edit-distance distribution per sample.

    Compared to the sequential version this reduces model calls from
    N × (1 + n_samples) to ceil(N / batch_size) × (1 + n_samples).
    """
    config = CDDConfig(n_samples=n_samples, max_token_length=max_new_tokens)
    detector = CDDDetector(
        generate_fn=None,
        tokenize_fn=lambda t: tokenizer(t, add_special_tokens=False)["input_ids"],
        config=config,
    )

    prompts = [_clean(t[:prompt_chars]) for t in texts]
    n = len(prompts)

    greedy_completions: List[str] = [""] * n
    stochastic_completions: List[List[str]] = [[] for _ in range(n)]

    # ── greedy pass ───────────────────────────────────────────────────────────
    logger.info(f"  CDD greedy pass ({n} prompts, batch={batch_size})…")
    for start in range(0, n, batch_size):
        batch = prompts[start:start + batch_size]
        results = _batch_generate(batch, model, tokenizer, device,
                                  max_new_tokens, do_sample=False)
        for j, r in enumerate(results):
            greedy_completions[start + j] = r
        if (start // batch_size + 1) % 10 == 0:
            logger.info(f"    greedy: {start + len(batch)}/{n}")

    # ── stochastic passes ─────────────────────────────────────────────────────
    logger.info(f"  CDD stochastic passes ({n_samples} × batch={batch_size})…")
    for s in range(n_samples):
        for start in range(0, n, batch_size):
            batch = prompts[start:start + batch_size]
            results = _batch_generate(batch, model, tokenizer, device,
                                      max_new_tokens, do_sample=True, temperature=0.8)
            for j, r in enumerate(results):
                stochastic_completions[start + j].append(r)
        if (s + 1) % 5 == 0:
            logger.info(f"    stochastic sample {s + 1}/{n_samples} done")

    # ── score ─────────────────────────────────────────────────────────────────
    rows = []
    for idx in range(n):
        score, is_mem = detector.score_from_completions(
            greedy=greedy_completions[idx],
            stochastic_samples=stochastic_completions[idx],
        )
        rows.append({
            "sample_idx": idx,
            "cdd_score": round(score, 6),
            "cdd_memorized": int(is_mem),
        })
    return rows


# ── CoDeC (batched) ───────────────────────────────────────────────────────────

def run_codec(
    texts: List[str],
    model_handler: ModelHandler,
    n_ctx: int,
    batch_size: int,
) -> List[dict]:
    """
    Batched CoDeC.

    Strategy:
      1. Pre-build per-sample context prefix strings.
      2. One batched forward pass for all `target_text` → lp_no_ctx[i].
      3. One batched forward pass for all `ctx + target_text` → lp_with_ctx[i].
      4. For each sample, extract the last len(lp_no_ctx[i]) elements from
         lp_with_ctx[i] — these are the target-token log-probs conditioned on
         the context (identical to CoDeC.detect's lp_with_ctx[-n:] slice).
      5. Apply the CoDeC decision rule: contaminated if conf_no_ctx > conf_with_ctx.

    Because the two forward passes are fully batched, GPU utilisation is close
    to 100% for the log-prob computation step.
    """
    rng = random.Random(42)

    # ── build context-prefixed texts ──────────────────────────────────────────
    with_ctx_texts: List[str] = []
    for i, target in enumerate(texts):
        pool = texts[:i] + texts[i + 1:]
        ctx_examples = rng.sample(pool, min(n_ctx, len(pool)))
        prefix = "\n\n".join(ctx_examples) + "\n\n" if ctx_examples else ""
        with_ctx_texts.append(_clean(prefix + target))

    no_ctx_texts = [_clean(t) for t in texts]

    # ── batched forward passes ────────────────────────────────────────────────
    logger.info(f"  CoDeC no-context pass ({len(texts)} texts, batch={batch_size})…")
    lp_no_ctx = batched_logprobs(
        model_handler.model, model_handler.tokenizer,
        no_ctx_texts, model_handler.device, batch_size,
    )

    logger.info(f"  CoDeC with-context pass ({len(texts)} texts, batch={batch_size})…")
    lp_with_ctx = batched_logprobs(
        model_handler.model, model_handler.tokenizer,
        with_ctx_texts, model_handler.device, batch_size,
    )

    # ── apply decision rule ───────────────────────────────────────────────────
    detector = CoDeC()
    tok_start, tok_end = detector.token_range   # (10, -1)

    rows = []
    for idx in range(len(texts)):
        n = len(lp_no_ctx[idx])
        if n == 0:
            rows.append({"sample_idx": idx, "codec_score": None, "codec_memorized": None})
            continue

        # Mirror CoDeC.detect: lp_with_ctx[-n:] = last n log-probs = target tokens
        lp_target_from_ctx = lp_with_ctx[idx][-n:] if len(lp_with_ctx[idx]) >= n else lp_with_ctx[idx]

        end_eff = min(tok_end if tok_end >= 0 else n, n)
        if tok_start >= n - 1:
            rows.append({"sample_idx": idx, "codec_score": None, "codec_memorized": None})
            continue

        conf_no_ctx   = float(np.mean(lp_no_ctx[idx][tok_start:end_eff]))
        conf_with_ctx = float(np.mean(lp_target_from_ctx[tok_start:end_eff]))
        score = 1.0 if conf_no_ctx > conf_with_ctx else 0.0

        rows.append({
            "sample_idx": idx,
            "codec_score": score,
            "codec_memorized": int(score == 1.0),
        })

    return rows


# ── ACR (sequential — inherent) ───────────────────────────────────────────────

def _ensure_acr_path() -> bool:
    """
    Add the ACR related-work directory to sys.path so that
    `import prompt_optimization` works at call time (not just at acr.py
    import time, where the lazy import hasn't fired yet).

    Returns True if the directory was found, False otherwise.
    """
    # Candidate locations relative to SRC_DIR and its parents
    candidates = [
        SRC_DIR.parent / "Related_work" / "acr-memorization-main",
        SRC_DIR.parent.parent / "Related_work" / "acr-memorization-main",
        Path(__file__).resolve().parent.parent / "Related_work" / "acr-memorization-main",
    ]
    for cand in candidates:
        if cand.exists():
            if str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
                logger.info(f"ACR: added to sys.path → {cand}")
            return True
    logger.error(
        "ACR: could not find acr-memorization-main/. "
        "Set ACR_ROOT env var or place the repo at "
        f"{candidates[0]}. All ACR samples will be skipped."
    )
    return False


def run_acr(
    texts: List[str],
    model,
    tokenizer,
    device: str,
    acr_steps: int,
) -> List[dict]:
    if not _ensure_acr_path():
        return [{"sample_idx": i, "acr_ratio": None, "acr_memorized": None}
                for i in range(len(texts))]
    from baseline.acr import ACRDetector, ACRConfig
    detector = ACRDetector(model, tokenizer, ACRConfig(num_steps=acr_steps))
    rows = []
    for idx, text in enumerate(texts):
        try:
            ratio, is_mem = detector.is_memorized(text)
        except Exception as exc:
            logger.warning(f"ACR sample {idx} failed: {exc}")
            ratio, is_mem = float("nan"), None
        rows.append({
            "sample_idx": idx,
            "acr_ratio": round(ratio, 6) if not np.isnan(ratio) else None,
            "acr_memorized": int(is_mem) if is_mem is not None else None,
        })
        if (idx + 1) % 10 == 0:
            logger.info(f"  ACR: {idx + 1}/{len(texts)} done")
    return rows


# ── orchestration ─────────────────────────────────────────────────────────────

def _max_samples_for_split(split_name: str, args) -> Optional[int]:
    if split_name == "members" and args.max_samples_members is not None:
        return args.max_samples_members
    if split_name == "non_members" and args.max_samples_non_members is not None:
        return args.max_samples_non_members
    return args.max_samples


def apply_config_defaults(args) -> None:
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = SRC_DIR / cfg_path
    if not cfg_path.is_file():
        return
    bl = load_config(str(cfg_path)).get("baseline") or {}
    if args.max_samples is None and bl.get("max_samples_per_split") is not None:
        args.max_samples = int(bl["max_samples_per_split"])
    if args.data_members == _DEFAULT_DATA_MEMBERS and bl.get("data_members"):
        args.data_members = bl["data_members"]
    if args.data_non_members == _DEFAULT_DATA_NON_MEMBERS and bl.get("data_non_members"):
        args.data_non_members = bl["data_non_members"]


def parse_args():
    p = argparse.ArgumentParser(description="Run memorization baselines across checkpoints")
    p.add_argument("--config", default="config.yml",
                   help="YAML config; baseline.max_samples_per_split used if --max-samples omitted")
    p.add_argument("--model", required=True, metavar="PATH_OR_ID")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--split", default="both",
                   choices=["members", "non_members", "both"])
    p.add_argument("--methods", nargs="+", default=["cdd", "codec"],
                   choices=["cdd", "codec", "acr"])
    p.add_argument("--output-dir", default="results/pythia_70m")
    p.add_argument("--data-members", default=_DEFAULT_DATA_MEMBERS)
    p.add_argument("--data-non-members", default=_DEFAULT_DATA_NON_MEMBERS)
    p.add_argument(
        "--max-samples", type=int, default=None,
        help="Max records per data file (applies to each split unless overridden below)",
    )
    p.add_argument(
        "--max-samples-members", type=int, default=None,
        help="Max records from --data-members (overrides --max-samples for members)",
    )
    p.add_argument(
        "--max-samples-non-members", type=int, default=None,
        help="Max records from --data-non-members (overrides --max-samples for non-members)",
    )
    p.add_argument("--batch-size", type=int, default=8,
                   help="Samples per GPU batch for CDD and CoDeC (default: 8)")
    p.add_argument("--prompt-chars", type=int, default=256,
                   help="Chars used as CDD prompt prefix (default: 256)")
    p.add_argument("--cdd-n-samples", type=int, default=20,
                   help="Stochastic samples per CDD instance (default: 20)")
    p.add_argument("--cdd-max-tokens", type=int, default=100)
    p.add_argument("--codec-n-ctx", type=int, default=2,
                   help="Context examples per CoDeC query (default: 2)")
    p.add_argument("--acr-steps", type=int, default=200,
                   help="GCG steps for ACR (default: 200; use 50 for quick tests)")
    return p.parse_args()


def run_split(split_name: str, data_path: str, args, model_handler: ModelHandler) -> None:
    out_path = Path(args.output_dir) / f"run_{args.epoch}" / split_name / "baseline_results.csv"

    limit = _max_samples_for_split(split_name, args)
    logger.info(f"=== {split_name.upper()} | epoch={args.epoch} | methods={args.methods} ===")
    texts = load_data(data_path, limit)
    logger.info(f"  {len(texts)} texts loaded from {data_path}")

    method_rows: dict = {}

    if "cdd" in args.methods:
        logger.info("Running CDD (batched)…")
        method_rows["cdd"] = run_cdd(
            texts,
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            device=model_handler.device,
            prompt_chars=args.prompt_chars,
            n_samples=args.cdd_n_samples,
            max_new_tokens=args.cdd_max_tokens,
            batch_size=args.batch_size,
        )

    if "codec" in args.methods:
        logger.info("Running CoDeC (batched)…")
        method_rows["codec"] = run_codec(
            texts,
            model_handler=model_handler,
            n_ctx=args.codec_n_ctx,
            batch_size=args.batch_size,
        )

    if "acr" in args.methods:
        logger.info("Running ACR (sequential — GCG per sample)…")
        method_rows["acr"] = run_acr(
            texts,
            model=model_handler.model,
            tokenizer=model_handler.tokenizer,
            device=model_handler.device,
            acr_steps=args.acr_steps,
        )

    save_csv(merge_results(method_rows, len(texts)), out_path)


def main():
    args = parse_args()
    apply_config_defaults(args)
    os.chdir(SRC_DIR)

    if args.max_samples is not None:
        logger.info("Sample limit (default): %d per file", args.max_samples)
    if args.max_samples_members is not None:
        logger.info("Sample limit (members): %d", args.max_samples_members)
    if args.max_samples_non_members is not None:
        logger.info("Sample limit (non-members): %d", args.max_samples_non_members)

    ckpt = Path(args.model).expanduser()
    if not ckpt.is_absolute():
        ckpt = (SRC_DIR / ckpt).resolve()
    logger.info(f"Loading model: {ckpt}")
    model_handler = ModelHandler(
        model_name=str(ckpt),
        checkpoint_path=str(ckpt),
        verbose=True,
        repo_root=SRC_DIR,
    )

    splits = (
        [("members", args.data_members), ("non_members", args.data_non_members)]
        if args.split == "both"
        else [("members", args.data_members)]
        if args.split == "members"
        else [("non_members", args.data_non_members)]
    )

    for split_name, data_path in splits:
        run_split(split_name, data_path, args, model_handler)

    logger.info("Done.")


if __name__ == "__main__":
    main()
