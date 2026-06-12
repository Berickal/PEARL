"""
IPA / PEARL analysis over experiment results.

Implements the calibration and detection framework from
``paper/sections/problem_formulation.tex``:

  - Neighborhood score  S(x) = A({output_similarity over N(x)})
  - Memorization gap    γ = μ(S_G) − μ(S_E)
  - Z-score             M_score(u) = (μ(S_G) − S(u)) / σ(S_G)
  - IPA decision rule   flag if S(u) < μ(S_G) − τ·γ  (τ configurable)

Also reports detection counts at the Youden-optimal threshold (as in the
paper results tables), AUC per aggregation operator, and optional overlap
with MIA / CDD when those files exist under ``results/``.

Outputs (default: ``evaluation/reports/``)::

    ipa_metrics.csv           — per model / epoch / operator
    detection_at_youden.csv   — TP, FP, precision, recall at Youden J
    instance_scores.csv       — per-sample S and flags (A_mean, one epoch)
    SUMMARY.md                — human-readable report
    plots/*.png               — curves, comparisons, Pythia-410M score boxplots

Usage (from ``src_v3/``)::

    python -m evaluation.ipa_analysis
    python -m evaluation.ipa_analysis --models pythia_410m --epoch 10
    python -m evaluation.ipa_analysis --results-dir results --out-dir evaluation/reports
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_style import (
    COLOR_ANNOT,
    COLOR_MEMBER,
    COLOR_NONMEMBER,
    FS_ANNOT,
    FS_LEGEND_COMPACT,
    FS_SUBPLOT_TITLE,
    FS_SUPTITLE,
    FS_TICK,
    VENN_FOUR_SET_STYLES,
    VENN_MIA_STYLE,
    VENN_PEARL_STYLE,
    FIG_AUC_GAMMA_EPOCHS,
    FIG_CROSS_MODEL,
    FIG_MODEL_SCALING,
    FIG_STANDARD,
    LINE_WIDTH,
    MARKER_SIZE,
    apply_paper_rcparams,
    configure_model_size_xaxis,
    cross_model_bar_kwargs,
    legend as plot_legend,
    set_axis_labels,
    style_grid,
)

apply_paper_rcparams()
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

# ── paths ─────────────────────────────────────────────────────────────────────
_EVAL_DIR = Path(__file__).resolve().parent
SRC_DIR = _EVAL_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

DEFAULT_RESULTS = SRC_DIR / "results"
DEFAULT_OUT = _EVAL_DIR / "reports"
PLOT_DIR_NAME = "plots"

MODEL_ORDER = ["pythia_70m", "pythia_410m", "pythia_1.4b", "pythia_2.8b"]
MODEL_LABELS = {
    "pythia_70m": "Pythia-70M",
    "pythia_410m": "Pythia-410M",
    "pythia_1.4b": "Pythia-1.4B",
    "pythia_2.8b": "Pythia-2.8B",
}
MODEL_PARAMS_M = {"pythia_70m": 70, "pythia_410m": 410,
                  "pythia_1.4b": 1400, "pythia_2.8b": 2800}

RANGE_ORDER = [
    "[0.90-1.00]", "[0.80-0.90]", "[0.70-0.80]",
    "[0.60-0.70]", "[0.50-0.60]",
]

# (operator_name, agg_fn, default_lower_S_is_member)
# Orientation is auto-calibrated per run from μ(S_E) vs μ(S_G); defaults are hints
# for operators where higher S usually indicates members (memorization stability).
IPA_OPERATORS: List[Tuple[str, Callable[[List[float]], float], bool]] = [
    ("A_mean", lambda v: float(np.mean(v)), False),
    ("A_min", lambda v: float(np.min(v)), False),
    ("A_neg_var", lambda v: float(-np.var(v)), True),
    ("A_q10", lambda v: float(np.quantile(v, 0.10)), False),
    ("A_q25", lambda v: float(np.quantile(v, 0.25)), False),
    ("A_median", lambda v: float(np.median(v)), False),
]

OP_COLORS = {
    "A_mean": "#1f77b4", "A_min": "#d62728", "A_neg_var": "#2ca02c",
    "A_q10": "#ff7f0e", "A_q25": "#9467bd", "A_median": "#8c564b",
}

MIA_METHODS = [
    ("mia_loss", False, "MIA-loss"),
    ("mia_min_k", True, "MIA-min-K"),
    ("mia_neighborhood", True, "MIA-neighborhood"),
]

MIA_COLORS = {
    "MIA-loss": "#e377c2",
    "MIA-min-K": "#17becf",
    "MIA-neighborhood": "#bcbd22",
}

# Cross-model bar chart: PEARL + all MIA methods (CDD excluded)
CROSS_MODEL_AUC_SERIES = [
    ("ipa_auc", "PEARL (A_mean)"),
    ("mia_loss_auc", "MIA-loss"),
    ("mia_min_k_auc", "MIA-min-K"),
    ("mia_neighborhood_auc", "MIA-neighborhood"),
]

CROSS_MODEL_FLAGGED_SERIES = [
    ("ipa_flagged_members", "PEARL"),
    ("mia_loss_flagged_members", "MIA-loss"),
    ("mia_min_k_flagged_members", "MIA-min-K"),
    ("mia_neighborhood_flagged_members", "MIA-neighborhood"),
]

# Score-distribution boxplots (members vs non-members)
BOXPLOT_MODEL_DEFAULT = "pythia_410m"
BOXPLOT_EPOCHS_DEFAULT = (1, 10)
MEMBER_BOX_COLOR = "#f4a8a8"
NONMEMBER_BOX_COLOR = "#a8c8f4"

# AUC / γ vs model size (PEARL A_mean)
MODEL_SIZE_EPOCHS = (1, 10)
EPOCH_LINE_STYLES = {
    0: {"color": "#7f7f7f", "marker": "o", "label": "Epoch 0 (base)"},
    1: {"color": "#2ca02c", "marker": "s", "label": "Epoch 1"},
    10: {"color": "#d62728", "marker": "^", "label": "Epoch 10"},
}
def mia_method_slug(label: str) -> str:
    """'MIA-min-K' → 'mia_min_k' for column names."""
    return label.lower().replace("-", "_")


# ── data loading ──────────────────────────────────────────────────────────────

def discover_models(results_root: Path) -> List[str]:
    found = []
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or d.name == "logs":
            continue
        if any(d.glob("run_*/members/records.json")):
            found.append(d.name)
    ordered = [m for m in MODEL_ORDER if m in found]
    return ordered + [m for m in found if m not in ordered]


def discover_epochs(model_dir: Path) -> List[int]:
    epochs = []
    for p in model_dir.glob("run_*"):
        if not p.is_dir():
            continue
        try:
            ep = int(p.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if (p / "members" / "records.json").exists() and (
            p / "non_members" / "records.json"
        ).exists():
            epochs.append(ep)
    return sorted(epochs)


def load_records(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def ipa_scores_by_sample(
    records: List[dict], agg_fn: Callable[[List[float]], float]
) -> Dict[int, float]:
    by_sample: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        sim = r.get("output_similarity")
        if sim is not None:
            by_sample[int(r["sample_idx"])].append(float(sim))
    return {
        idx: agg_fn(vals)
        for idx, vals in by_sample.items()
        if vals
    }


def align_scores(
    scores_E: Dict[int, float], scores_G: Dict[int, float]
) -> Tuple[np.ndarray, np.ndarray, List[int], List[int]]:
    """Return parallel arrays for members (E) and non-members (G)."""
    idx_E = sorted(scores_E.keys())
    idx_G = sorted(scores_G.keys())
    return (
        np.array([scores_E[i] for i in idx_E], dtype=float),
        np.array([scores_G[i] for i in idx_G], dtype=float),
        idx_E,
        idx_G,
    )


# ── IPA metrics ───────────────────────────────────────────────────────────────

def memorization_score_z(s_u: float, mu_G: float, sigma_G: float) -> float:
    """M_score(u) = (μ(S_G) − S(u)) / σ(S_G)  (paper Eq. before Hypothesis)."""
    if sigma_G <= 1e-12:
        return 0.0
    return (mu_G - s_u) / sigma_G


@dataclass
class IPARunMetrics:
    model: str
    epoch: int
    operator: str
    n_members: int
    n_non_members: int
    mu_E: float
    mu_G: float
    sigma_E: float
    sigma_G: float
    gamma: float
    auc: float
    auc_raw: float
    members_higher_S: bool
    tau: float
    # Youden-optimal detection (members = positive class)
    youden_threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float
    precision: float
    flagged_members: int
    flagged_non_members: int


def members_higher_than_generalization(s_E: np.ndarray, s_G: np.ndarray) -> bool:
    """True when μ(S_E) > μ(S_G) — stable outputs on members (empirical PSH)."""
    return float(np.mean(s_E)) > float(np.mean(s_G))


def detection_score(s: float, members_higher_S: bool) -> float:
    """Higher score ⇒ more likely member (positive class for ROC / Youden)."""
    return s if members_higher_S else -s


def resolve_orientation(
    s_E: np.ndarray,
    s_G: np.ndarray,
    default_lower_S: bool,
) -> bool:
    """
    Return members_higher_S for AUC / detection.

    Uses the empirical gap direction; falls back to the operator default when
    |μ_E − μ_G| is negligible.
    """
    if abs(float(np.mean(s_E)) - float(np.mean(s_G))) < 1e-9:
        return not default_lower_S
    return members_higher_than_generalization(s_E, s_G)


def safe_auc(y_true: List[int], y_score: List[float]) -> float:
    if len(set(y_true)) < 2 or not y_score:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def youden_threshold(
    s_E: np.ndarray, s_G: np.ndarray, members_higher_S: bool
) -> Tuple[float, int, int, int, int]:
    """
    Youden J on combined labeled set (members positive).
    Returns (threshold_on_detection_score, tp, fp, fn, tn).
    """
    y_true = np.concatenate([np.ones(len(s_E)), np.zeros(len(s_G))])
    y_score = np.concatenate([
        [detection_score(s, members_higher_S) for s in s_E],
        [detection_score(s, members_higher_S) for s in s_G],
    ])
    if len(s_E) == 0 or len(s_G) == 0:
        return float("nan"), 0, 0, 0, 0

    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    j = tpr - fpr
    best = int(np.argmax(j))
    thr = float(thresholds[best])
    pred = y_score >= thr
    tp = int(np.sum(pred[: len(s_E)]))
    fn = len(s_E) - tp
    fp = int(np.sum(pred[len(s_E) :]))
    tn = len(s_G) - fp
    return thr, tp, fp, fn, tn


def ipa_rule_flags(
    s_E: np.ndarray,
    s_G: np.ndarray,
    mu_G: float,
    gamma: float,
    tau: float,
    members_higher_S: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    IPA decision rule (paper Alg. 1), both orientations.

    Classic (members more brittle):  S(u) < μ(S_G) − τ·γ
    Empirical PSH (members stable):  S(u) > μ(S_G) − τ·γ
    """
    cutoff = mu_G - tau * gamma
    if members_higher_S:
        return s_E > cutoff, s_G > cutoff
    return s_E < cutoff, s_G < cutoff


def analyse_ipa_run(
    model: str,
    epoch: int,
    records_E: List[dict],
    records_G: List[dict],
    operator: str,
    agg_fn: Callable[[List[float]], float],
    default_lower_S: bool,
    tau: float = 1.0,
) -> Optional[IPARunMetrics]:
    scores_E = ipa_scores_by_sample(records_E, agg_fn)
    scores_G = ipa_scores_by_sample(records_G, agg_fn)
    if not scores_E or not scores_G:
        return None

    s_E, s_G, _, _ = align_scores(scores_E, scores_G)
    mu_E, mu_G = float(np.mean(s_E)), float(np.mean(s_G))
    sigma_E, sigma_G = float(np.std(s_E)), float(np.std(s_G))
    gamma = mu_G - mu_E  # paper Def.: positive ⇒ members more brittle (lower S)
    memb_hi = resolve_orientation(s_E, s_G, default_lower_S)

    y_true = [1] * len(s_E) + [0] * len(s_G)
    y_score = (
        [detection_score(s, memb_hi) for s in s_E]
        + [detection_score(s, memb_hi) for s in s_G]
    )
    auc_raw = safe_auc(y_true, [-s for s in s_E] + [-s for s in s_G])
    auc = safe_auc(y_true, y_score)

    thr, tp, fp, fn, tn = youden_threshold(s_E, s_G, memb_hi)
    recall = tp / len(s_E) if len(s_E) else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    rule_E, rule_G = ipa_rule_flags(s_E, s_G, mu_G, gamma, tau, memb_hi)
    flagged_E = int(np.sum(rule_E))
    flagged_G = int(np.sum(rule_G))

    return IPARunMetrics(
        model=model,
        epoch=epoch,
        operator=operator,
        n_members=len(s_E),
        n_non_members=len(s_G),
        mu_E=mu_E,
        mu_G=mu_G,
        sigma_E=sigma_E,
        sigma_G=sigma_G,
        gamma=gamma,
        auc=auc,
        auc_raw=auc_raw,
        members_higher_S=memb_hi,
        tau=tau,
        youden_threshold=thr,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        recall=recall,
        precision=precision,
        flagged_members=flagged_E,
        flagged_non_members=flagged_G,
    )


def metrics_to_row(m: IPARunMetrics) -> dict:
    return {
        "model": m.model,
        "model_label": MODEL_LABELS.get(m.model, m.model),
        "params_M": MODEL_PARAMS_M.get(m.model),
        "epoch": m.epoch,
        "operator": m.operator,
        "n_members": m.n_members,
        "n_non_members": m.n_non_members,
        "mu_S_E": round(m.mu_E, 6),
        "mu_S_G": round(m.mu_G, 6),
        "sigma_S_E": round(m.sigma_E, 6),
        "sigma_S_G": round(m.sigma_G, 6),
        "gamma": round(m.gamma, 6),
        "abs_gamma": round(abs(m.gamma), 6),
        "members_higher_S": m.members_higher_S,
        "auc": round(m.auc, 4),
        "auc_raw_neg_S": round(m.auc_raw, 4),
        "tau": m.tau,
        "youden_threshold": round(m.youden_threshold, 6)
        if not np.isnan(m.youden_threshold)
        else None,
        "tp": m.tp,
        "fp": m.fp,
        "fn": m.fn,
        "tn": m.tn,
        "recall": round(m.recall, 4),
        "precision": round(m.precision, 4),
        "flagged_members_ipa_rule": m.flagged_members,
        "flagged_non_members_ipa_rule": m.flagged_non_members,
        "memorized_members_youden": m.tp,
        "memorized_non_members_youden": m.fp,
    }


# ── MIA / CDD helpers ─────────────────────────────────────────────────────────

def analyse_mia(
    model: str, epoch: int, run_dir: Path
) -> List[dict]:
    rows = []
    mia_E = run_dir / "members" / "mia_summary.csv"
    mia_G = run_dir / "non_members" / "mia_summary.csv"
    if not mia_E.exists() or not mia_G.exists():
        return rows
    df_E = pd.read_csv(mia_E)
    df_G = pd.read_csv(mia_G)
    for col, higher_mem, label in MIA_METHODS:
        if col not in df_E.columns or col not in df_G.columns:
            continue
        s_E = df_E[col].dropna().astype(float).values
        s_G = df_G[col].dropna().astype(float).values
        if len(s_E) == 0 or len(s_G) == 0:
            continue
        y_true = [1] * len(s_E) + [0] * len(s_G)
        raw = np.concatenate([s_E, s_G])
        y_score = raw if higher_mem else -raw
        auc = safe_auc(y_true, y_score.tolist())
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        j = tpr - fpr
        best = int(np.argmax(j))
        thr = float(thresholds[best])
        pred = y_score >= thr
        tp = int(np.sum(pred[: len(s_E)]))
        fp = int(np.sum(pred[len(s_E) :]))
        rows.append({
            "model": model,
            "epoch": epoch,
            "method": label,
            "method_col": col,
            "auc": round(auc, 4),
            "tp": tp,
            "fp": fp,
            "recall": round(tp / len(s_E), 4),
            "precision": round(tp / (tp + fp), 4) if (tp + fp) else 0.0,
            "memorized_members": tp,
            "memorized_non_members": fp,
        })
    return rows


def analyse_cdd(model: str, epoch: int, run_dir: Path) -> Optional[dict]:
    p_E = run_dir / "members" / "baseline_results.csv"
    p_G = run_dir / "non_members" / "baseline_results.csv"
    if not p_E.exists() or not p_G.exists():
        return None
    df_E = pd.read_csv(p_E)
    df_G = pd.read_csv(p_G)
    if "cdd_score" not in df_E.columns:
        return None
    s_E = df_E["cdd_score"].dropna().astype(float).values
    s_G = df_G["cdd_score"].dropna().astype(float).values
    y_true = [1] * len(s_E) + [0] * len(s_G)
    auc = safe_auc(y_true, np.concatenate([s_E, s_G]).tolist())
    mem_E = int(df_E["cdd_memorized"].sum()) if "cdd_memorized" in df_E.columns else None
    mem_G = int(df_G["cdd_memorized"].sum()) if "cdd_memorized" in df_G.columns else None
    return {
        "model": model,
        "epoch": epoch,
        "method": "CDD",
        "auc": round(auc, 4),
        "mu_S_E": round(float(np.mean(s_E)), 4),
        "mu_S_G": round(float(np.mean(s_G)), 4),
        "gamma": round(float(np.mean(s_E) - np.mean(s_G)), 4),
        "memorized_members": mem_E,
        "memorized_non_members": mem_G,
    }


def overlap_jaccard(flags_a: np.ndarray, flags_b: np.ndarray) -> float:
    both = int(np.sum(flags_a & flags_b))
    either = int(np.sum(flags_a | flags_b))
    return both / either if either else 0.0


def mia_youden_flags_members(
    run_dir: Path, col: str, higher_mem: bool
) -> Optional[Tuple[np.ndarray, List[int]]]:
    """Youden flags on members only, aligned to sample_idx order in members CSV."""
    mia_E = run_dir / "members" / "mia_summary.csv"
    mia_G = run_dir / "non_members" / "mia_summary.csv"
    if not mia_E.exists() or not mia_G.exists():
        return None
    df_E = pd.read_csv(mia_E)
    df_G = pd.read_csv(mia_G)
    if col not in df_E.columns or col not in df_G.columns:
        return None
    s_E = df_E[col].astype(float).values
    s_G = df_G[col].astype(float).values
    y_E = s_E if higher_mem else -s_E
    y_G = s_G if higher_mem else -s_G
    y_true = np.concatenate([np.ones(len(y_E)), np.zeros(len(y_G))])
    combined = np.concatenate([y_E, y_G])
    fpr, tpr, thresholds = roc_curve(y_true, combined)
    thr = float(thresholds[int(np.argmax(tpr - fpr))])
    flags_E = y_E >= thr
    return flags_E, df_E["sample_idx"].astype(int).tolist()


def compute_mia_overlap(
    model: str,
    epoch: int,
    run_dir: Path,
    ipa_flags_E: np.ndarray,
    sample_indices_E: List[int],
) -> List[dict]:
    """Pairwise overlap: PEARL (A_mean, Youden) vs each MIA on members."""
    mia_path = run_dir / "members" / "mia_summary.csv"
    if not mia_path.exists():
        return []
    df = pd.read_csv(mia_path)
    if "sample_idx" not in df.columns:
        return []

    ipa_by_idx = {idx: bool(f) for idx, f in zip(sample_indices_E, ipa_flags_E)}
    rows: List[dict] = []
    n_members = len(df)

    for col, higher_mem, label in MIA_METHODS:
        got = mia_youden_flags_members(run_dir, col, higher_mem)
        if got is None:
            continue
        mia_flags, mia_indices = got
        ipa_vec = np.array([ipa_by_idx.get(i, False) for i in mia_indices])
        if len(ipa_vec) != len(mia_flags):
            continue

        both = int(np.sum(ipa_vec & mia_flags))
        pearl_only = int(np.sum(ipa_vec & ~mia_flags))
        mia_only = int(np.sum(~ipa_vec & mia_flags))
        neither = int(n_members - both - pearl_only - mia_only)
        pearl_flagged = both + pearl_only
        mia_flagged = both + mia_only

        rows.append({
            "model": model,
            "epoch": epoch,
            "mia_method": label,
            "n_members": n_members,
            "pearl_flagged": pearl_flagged,
            "mia_flagged": mia_flagged,
            "both": both,
            "pearl_only": pearl_only,
            "mia_only": mia_only,
            "neither": neither,
            "jaccard": round(overlap_jaccard(ipa_vec, mia_flags), 4),
        })
    return rows


def gather_pearl_mia_member_flags(
    run_dir: Path,
    ipa_flags_E: np.ndarray,
    sample_indices_E: List[int],
) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray], int]]:
    """Aligned member-level Youden flags: PEARL + each MIA method."""
    mia_path = run_dir / "members" / "mia_summary.csv"
    if not mia_path.exists():
        return None
    df = pd.read_csv(mia_path)
    if "sample_idx" not in df.columns:
        return None

    ipa_by_idx = {idx: bool(f) for idx, f in zip(sample_indices_E, ipa_flags_E)}
    mia_flags: Dict[str, np.ndarray] = {}
    ref_indices: Optional[List[int]] = None

    for col, higher_mem, label in MIA_METHODS:
        got = mia_youden_flags_members(run_dir, col, higher_mem)
        if got is None:
            continue
        flags, indices = got
        mia_flags[label] = flags
        ref_indices = indices

    if not mia_flags or ref_indices is None:
        return None

    pearl = np.array([ipa_by_idx.get(i, False) for i in ref_indices], dtype=bool)
    return pearl, mia_flags, len(ref_indices)


def count_pearl_mia_regions(
    pearl: np.ndarray,
    mia_flags: Dict[str, np.ndarray],
) -> Dict[Tuple[int, ...], int]:
    """Count members in each PEARL × MIA-loss × MIA-min-K × MIA-neighborhood region."""
    methods = [label for _, _, label in MIA_METHODS if label in mia_flags]
    counts: Dict[Tuple[int, ...], int] = defaultdict(int)
    n = len(pearl)
    for i in range(n):
        key = (int(bool(pearl[i])),) + tuple(int(bool(mia_flags[m][i])) for m in methods)
        counts[key] += 1
    return dict(counts)


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_auc_gamma_epochs(df_ipa: pd.DataFrame, model: str, out_dir: Path) -> None:
    sub = df_ipa[df_ipa["model"] == model]
    if sub.empty:
        return
    label = MODEL_LABELS.get(model, model)
    epochs = sorted(sub["epoch"].unique())
    fig, axes = plt.subplots(1, 2, figsize=FIG_AUC_GAMMA_EPOCHS)

    for op, _, _ in IPA_OPERATORS:
        s = sub[sub["operator"] == op].sort_values("epoch")
        if s.empty:
            continue
        axes[0].plot(
            s["epoch"], s["auc"], "o-", label=op,
            color=OP_COLORS.get(op, "grey"), linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        )
        axes[1].plot(
            s["epoch"], s["abs_gamma"], "o-", label=op,
            color=OP_COLORS.get(op, "grey"), linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
        )

    axes[0].axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5)
    set_axis_labels(
        axes[0],
        xlabel="Epoch",
        ylabel="AUC (members positive)",
        title=f"IPA detection AUC — {label}",
    )
    axes[0].set_xticks(epochs)
    plot_legend(axes[0], loc="best")
    style_grid(axes[0])

    axes[1].axhline(0, color="k", ls="--", lw=0.8, alpha=0.5)
    set_axis_labels(
        axes[1],
        xlabel="Epoch",
        ylabel="|γ| = |μ(S_G) − μ(S_E)|",
        title=f"Memorization gap — {label}",
    )
    axes[1].set_xticks(epochs)
    plot_legend(axes[1], loc="best")
    style_grid(axes[1])

    plt.suptitle(f"IPA calibration metrics — {label}", fontsize=FS_SUPTITLE)
    plt.tight_layout()
    fig.savefig(out_dir / f"{model}_auc_gamma_epochs.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_flagged_counts(
    df_det: pd.DataFrame, model: str, operator: str, out_dir: Path
) -> None:
    sub = df_det[
        (df_det["model"] == model) & (df_det["operator"] == operator)
    ].sort_values("epoch")
    if sub.empty:
        return
    label = MODEL_LABELS.get(model, model)
    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    ax.plot(
        sub["epoch"], sub["memorized_members_youden"], "o-",
        color=COLOR_MEMBER, label="Flagged members (TP)", linewidth=LINE_WIDTH,
    )
    ax.plot(
        sub["epoch"], sub["memorized_non_members_youden"], "s--",
        color=COLOR_NONMEMBER, label="Flagged non-members (FP)", linewidth=LINE_WIDTH,
    )
    set_axis_labels(
        ax,
        xlabel="Epoch",
        ylabel="# instances flagged (Youden J)",
        title=f"IPA {operator} — {label}",
    )
    plot_legend(ax)
    style_grid(ax)
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model}_{operator}_flagged_youden.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_score_distributions(
    s_E: np.ndarray,
    s_G: np.ndarray,
    model: str,
    epoch: int,
    operator: str,
    mu_E: float,
    mu_G: float,
    gamma: float,
    tau: float,
    out_dir: Path,
) -> None:
    label = MODEL_LABELS.get(model, model)
    memb_hi = mu_E > mu_G
    cutoff = mu_G - tau * gamma
    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    ax.hist(s_G, bins=40, alpha=0.55, density=True, label="Non-members (G)", color=COLOR_NONMEMBER)
    ax.hist(s_E, bins=40, alpha=0.55, density=True, label="Members (E)", color=COLOR_MEMBER)
    ax.axvline(mu_E, color=COLOR_MEMBER, ls="--", lw=1.2, label=f"μ(S_E)={mu_E:.3f}")
    ax.axvline(mu_G, color=COLOR_NONMEMBER, ls="--", lw=1.2, label=f"μ(S_G)={mu_G:.3f}")
    rule_lbl = "S > cutoff" if memb_hi else "S < cutoff"
    ax.axvline(cutoff, color="k", ls="-.", lw=1.5,
               label=f"IPA cutoff (τ={tau}, {rule_lbl})")
    set_axis_labels(
        ax,
        xlabel=f"Neighborhood score S(x) — {operator}",
        ylabel="Density",
        title=f"Score distributions — {label}, epoch {epoch}",
    )
    plot_legend(ax, compact=True)
    style_grid(ax)
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model}_epoch{epoch}_{operator}_score_dist.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_similarity_curves(
    run_dir: Path, model: str, epoch: int, out_dir: Path
) -> None:
    """Plot mean output similarity vs input bucket (members vs non-members)."""
    paths = {
        "Members": run_dir / "members" / "summary.csv",
        "Non-members": run_dir / "non_members" / "summary.csv",
    }
    series = {}
    for name, p in paths.items():
        if not p.exists():
            continue
        df = pd.read_csv(p)
        if "range" not in df.columns or "mean_output_sim" not in df.columns:
            continue
        df = df.set_index("range")
        series[name] = [df.loc[r, "mean_output_sim"] if r in df.index else np.nan
                        for r in RANGE_ORDER]

    if not series:
        return

    label = MODEL_LABELS.get(model, model)
    x = np.arange(len(RANGE_ORDER))
    fig, ax = plt.subplots(figsize=FIG_STANDARD)
    for name, vals in series.items():
        ax.plot(
            x, vals, "o-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            label=name,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(RANGE_ORDER, rotation=25, ha="right", fontsize=FS_TICK)
    set_axis_labels(
        ax,
        xlabel="Input similarity bucket",
        ylabel="Mean output similarity",
        title=f"Perturbation sensitivity curve — {label}, epoch {epoch}",
    )
    plot_legend(ax)
    style_grid(ax)
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model}_epoch{epoch}_similarity_curve.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_cross_model_bars(df_comp: pd.DataFrame, epoch: int, out_dir: Path) -> None:
    """Grouped AUC: PEARL + MIA-loss + MIA-min-K + MIA-neighborhood (no CDD)."""
    sub = df_comp[df_comp["epoch"] == epoch].sort_values("params_M")
    if sub.empty:
        return

    series = [(col, title) for col, title in CROSS_MODEL_AUC_SERIES if col in sub.columns]
    if not series:
        return

    models = sub["model_label"].tolist()
    n_m = len(models)
    n_s = len(series)
    x = np.arange(n_m)
    w = 0.8 / n_s
    offset_start = -(n_s - 1) / 2 * w

    fig, ax = plt.subplots(figsize=(max(FIG_CROSS_MODEL[0], 2.2 * n_m), FIG_CROSS_MODEL[1]))
    for i, (col, title) in enumerate(series):
        pos = x + offset_start + i * w
        vals = sub[col].values.astype(float)
        bars = ax.bar(
            pos, vals, w * 0.92, label=title,
            **cross_model_bar_kwargs(i),
        )
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 0.006,
                    f"", ha="center", va="bottom", fontsize=FS_ANNOT, rotation=0,
                )

    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5, label="Random (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=FS_TICK)
    set_axis_labels(
        ax,
        ylabel="AUC Score",
        title=(
            f"Cross-model detection AUC — epoch {epoch}"
            ""
        ),
    )
    ax.set_ylim(0.4, 0.88)
    plot_legend(ax, compact=True, loc="upper left", ncol=2, fontsize=15)
    style_grid(ax, axis="y")
    plt.tight_layout()
    fig.savefig(out_dir / f"cross_model_auc_epoch{epoch}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cross_model_flagged_bars(df_comp: pd.DataFrame, epoch: int, out_dir: Path) -> None:
    """Flagged member counts (memorization / membership) per method."""
    sub = df_comp[df_comp["epoch"] == epoch].sort_values("params_M")
    if sub.empty:
        return

    series = [(col, title) for col, title in CROSS_MODEL_FLAGGED_SERIES if col in sub.columns]
    if not series:
        return

    models = sub["model_label"].tolist()
    n_m = len(models)
    n_s = len(series)
    x = np.arange(n_m)
    w = 0.8 / n_s
    offset_start = -(n_s - 1) / 2 * w

    fig, ax = plt.subplots(figsize=(max(FIG_CROSS_MODEL[0], 2.2 * n_m), FIG_CROSS_MODEL[1]))
    for i, (col, title) in enumerate(series):
        pos = x + offset_start + i * w
        vals = sub[col].values.astype(float)
        bars = ax.bar(
            pos, vals, w * 0.92, label=title,
            **cross_model_bar_kwargs(i),
        )
        for bar, v in zip(bars, vals):
            if pd.notna(v):
                ax.text(
                    bar.get_x() + bar.get_width() / 2, v + 8,
                    f"", ha="center", va="bottom", fontsize=12,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=FS_TICK)
    set_axis_labels(
        ax,
        ylabel="# flagged members / memorized",
        title=f"Cross-model flagged member counts — epoch {epoch}",
    )
    plot_legend(ax, compact=True, loc="upper left", ncol=2, fontsize=15)
    style_grid(ax, axis="y")
    plt.tight_layout()
    fig.savefig(
        out_dir / f"cross_model_flagged_epoch{epoch}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.close(fig)


def plot_venn2(
    ax: plt.Axes,
    pearl_only: int,
    both: int,
    mia_only: int,
    neither: int,
    label_pearl: str = "PEARL",
    label_mia: str = "MIA",
    *,
    large: bool = False,
) -> None:
    """Draw a two-set Venn diagram on the given axes (grayscale + hatch)."""
    from matplotlib.patches import Circle

    ax.set_aspect("equal")
    ax.axis("off")

    r = 0.58 if large else 0.52
    fs_count = 20 if large else 13
    fs_label = 14 if large else 10
    fs_caption = 11 if large else 8
    fs_neither = 12 if large else 9

    pearl_s = VENN_PEARL_STYLE
    mia_s = VENN_MIA_STYLE
    ax.add_patch(Circle(
        (-0.32, 0), r,
        fc=pearl_s["facecolor"], alpha=0.9,
        ec=pearl_s["edgecolor"], lw=pearl_s["lw"], hatch=pearl_s["hatch"],
    ))
    ax.add_patch(Circle(
        (0.32, 0), r,
        fc=mia_s["facecolor"], alpha=0.9,
        ec=mia_s["edgecolor"], lw=mia_s["lw"], hatch=mia_s["hatch"],
    ))

    ax.text(-0.58, 0, str(pearl_only), ha="center", va="center",
            fontsize=fs_count, fontweight="bold", color="#222")
    ax.text(0, 0, str(both), ha="center", va="center",
            fontsize=fs_count, fontweight="bold", color="#222")
    ax.text(0.58, 0, str(mia_only), ha="center", va="center",
            fontsize=fs_count, fontweight="bold", color="#222")

    ax.text(-0.55, 0.78, label_pearl, ha="center", fontsize=fs_label,
            color=pearl_s["edgecolor"], fontweight="bold")
    ax.text(0.55, 0.78, label_mia, ha="center", fontsize=fs_label,
            color=mia_s["edgecolor"], fontweight="bold")

    ax.text(-0.32, -0.46, "PEARL only", ha="center", fontsize=fs_caption, color="#444")
    ax.text(0, -0.46, "Both", ha="center", fontsize=fs_caption, color="#444")
    ax.text(0.32, -0.46, f"{label_mia} only", ha="center", fontsize=fs_caption, color="#444")

    if neither > 0:
        ax.text(0, -0.86, f"Neither flagged: {neither}",
                ha="center", fontsize=fs_neither, style="italic", color="#555")

    lim = 1.25 if large else 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-1.05 if large else -0.95, 1.05 if large else 0.95)


# ── 4-set Venn geometry ───────────────────────────────────────────────────────
# Circles: PEARL(-0.36,0,r=0.60) | MIA-loss(0.16,0.40,r=0.55)
#          MIA-min-K(0.34,0,r=0.56) | MIA-neigh(0.16,-0.40,r=0.57)
# All four circles overlap at the centre (~0.10, 0.00).
# Radii are proportional to sqrt(set_size / PEARL_size).
#
# Label positions are verified to lie inside the correct intersection region.
# Small regions use leader lines (see _LEADER_REGIONS in the function).
_VENN4_LABEL_POS: Dict[Tuple[int, ...], Tuple[float, float]] = {
    (1, 0, 0, 0): (-0.82, 0.00),   # PEARL only            — far left
    (0, 1, 0, 0): ( 0.16, 0.90),   # MIA-loss only         — top
    (0, 0, 1, 0): ( 0.90, 0.00),   # MIA-min-K only        — far right
    (0, 0, 0, 1): ( 0.16,-0.90),   # MIA-neigh only        — bottom
    (1, 1, 0, 0): (-0.10, 0.44),   # PEARL ∩ MIA-loss      — upper-left (leader)
    (1, 0, 1, 0): (-0.05,-0.04),   # PEARL ∩ MIA-min-K     — thin sliver (leader)
    (1, 0, 0, 1): (-0.10,-0.44),   # PEARL ∩ MIA-neigh     — lower-left
    (0, 1, 1, 0): ( 0.44, 0.28),   # MIA-loss ∩ MIA-min-K  — upper-right
    (0, 1, 0, 1): ( 0.28, 0.00),   # MIA-loss ∩ MIA-neigh  — (leader)
    (0, 0, 1, 1): ( 0.44,-0.28),   # MIA-min-K ∩ MIA-neigh — lower-right
    (1, 1, 1, 0): (-0.02, 0.28),   # PEARL ∩ loss ∩ min-K  — upper-centre
    (1, 1, 0, 1): (-0.02,-0.28),   # PEARL ∩ loss ∩ neigh  — lower-centre
    (1, 0, 1, 1): ( 0.20, 0.00),   # PEARL ∩ min-K ∩ neigh — centre-right (leader)
    (0, 1, 1, 1): ( 0.48, 0.00),   # MIA triple            — right-centre
    (1, 1, 1, 1): ( 0.10, 0.00),   # All four              — centre
    (0, 0, 0, 0): (-0.92,-0.78),   # Neither               — bottom-left corner
}

# Colour palette — proportional, accessible, high-contrast
_VENN4_COLORS = [
    {"fc": "#2166ac", "ec": "#053061", "label": "PEARL"},            # deep blue
    {"fc": "#d6604d", "ec": "#8b1a0e", "label": "MIA-loss"},         # warm red
    {"fc": "#35978f", "ec": "#01665e", "label": "MIA-min-K"},        # teal
    {"fc": "#f4a582", "ec": "#b35806", "label": "MIA-neighborhood"}, # amber
]
_VENN4_ALPHA = 0.28


def plot_pearl_all_mias_unified_venn(
    pearl: np.ndarray,
    mia_flags: Dict[str, np.ndarray],
    model: str,
    epoch: int,
    out_dir: Path,
) -> None:
    """Single large 4-set Venn: PEARL together with all MIA methods (enhanced)."""
    from matplotlib.patches import Circle, Patch

    methods = [label for _, _, label in MIA_METHODS if label in mia_flags]
    if len(methods) < 2:
        return

    counts = count_pearl_mia_regions(pearl, mia_flags)
    n_members = len(pearl)

    # ── proportional geometry ─────────────────────────────────────────────────
    # Radii ∝ sqrt(set_size) so areas are proportional to flagged counts.
    # All four circles overlap at the centre so the dominant all-four region
    # (typically the largest) is visually prominent.
    set_sizes = {"PEARL": int(np.sum(pearl))}
    for m in methods:
        set_sizes[m] = int(np.sum(mia_flags[m]))
    r_pearl = 0.60
    radii = [
        r_pearl * np.sqrt(set_sizes.get("PEARL", 609) / max(set_sizes.values())),
        r_pearl * np.sqrt(set_sizes.get("MIA-loss", 514) / max(set_sizes.values())),
        r_pearl * np.sqrt(set_sizes.get("MIA-min-K", 540) / max(set_sizes.values())),
        r_pearl * np.sqrt(set_sizes.get("MIA-neighborhood", 553) / max(set_sizes.values())),
    ]
    # Clamp so largest circle is exactly r_pearl
    r_scale = r_pearl / max(radii)
    radii = [r * r_scale for r in radii]

    circles = [
        ((-0.36, 0.00), radii[0]),   # PEARL      — large, left
        (( 0.16, 0.40), radii[1]),   # MIA-loss   — top-right
        (( 0.34, 0.00), radii[2]),   # MIA-min-K  — right
        (( 0.16,-0.40), radii[3]),   # MIA-neigh  — bottom-right
    ]

    # Small / cramped interior regions → annotate outside with leader lines
    _LEADER_REGIONS: Dict[Tuple[int, ...], Tuple[Tuple[float, float], Tuple[float, float]]] = {
        (1, 1, 0, 0): ((-0.10,  0.44), (-0.95,  0.65)),
        (1, 0, 1, 0): ((-0.05, -0.04), (-0.95, -0.30)),
        (0, 1, 0, 1): (( 0.28,  0.00), ( 1.00,  0.38)),
        (1, 0, 1, 1): (( 0.20,  0.00), ( 1.00, -0.38)),
        (1, 1, 0, 1): ((-0.02, -0.28), (-0.90, -0.65)),
    }

    fig, ax = plt.subplots(figsize=(13, 11))
    ax.set_aspect("equal")
    ax.axis("off")

    # Draw circles (back to front: PEARL largest → behind MIA circles)
    for i, ((cx, cy), r) in enumerate(circles):
        col = _VENN4_COLORS[i]
        ax.add_patch(Circle(
            (cx, cy), r,
            fc=col["fc"], alpha=_VENN4_ALPHA,
            ec=col["ec"], lw=2.8, zorder=2,
        ))

    # ── count labels (no percentage) ─────────────────────────────────────────
    for key, count in counts.items():
        pos = _VENN4_LABEL_POS.get(key)
        if pos is None or count <= 0:
            continue

        # "Neither" region — small italic note, no box
        if key == (0, 0, 0, 0):
            ax.text(
                pos[0], pos[1],
                f"Neither: {count}",
                ha="center", va="center", fontsize=10,
                style="italic", color="#888888",
                zorder=6,
            )
            continue

        label_str = str(count)

        if key in _LEADER_REGIONS:
            inner, outer = _LEADER_REGIONS[key]
            ax.annotate(
                label_str,
                xy=inner, xytext=outer,
                ha="center", va="center",
                fontsize=12, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.28", fc="white",
                          ec="#aaaaaa", alpha=0.92, lw=1.0),
                arrowprops=dict(arrowstyle="-", color="#999999",
                                lw=1.2, shrinkA=3, shrinkB=3),
                zorder=7,
            )
        else:
            ax.text(
                pos[0], pos[1], label_str,
                ha="center", va="center",
                fontsize=15, fontweight="bold", color="#111111",
                zorder=6,
            )

    # ── set name labels (outside circles, aligned to new centres) ────────────
    label_offsets = [
        ((-0.36,  radii[0] + 0.10), "center"),  # PEARL
        (( 0.16,  radii[1] + 0.50), "center"),  # MIA-loss
        (( 0.34 + radii[2] + 0.08, 0.00), "left"),  # MIA-min-K
        (( 0.16, -(radii[3] + 0.50)), "center"), # MIA-neigh
    ]
    for i, ((tx, ty), ha) in enumerate(label_offsets):
        col = _VENN4_COLORS[i]
        ax.text(tx, ty, col["label"],
                ha=ha, va="center",
                fontsize=14.5, fontweight="bold",
                color=col["ec"], zorder=7)

    # ── legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        Patch(fc=c["fc"], ec=c["ec"], alpha=0.70, lw=1.8, label=c["label"])
        for c in _VENN4_COLORS
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left", bbox_to_anchor=(-0.02, 1.05),
        framealpha=0.94, fontsize=11.5,
        title="Detection method", title_fontsize=10.5,
        edgecolor="#cccccc",
    )

    # ── titles ────────────────────────────────────────────────────────────────
    label_str = MODEL_LABELS.get(model, model)
    fig.suptitle(
        f"PEARL ∩ MIA — {label_str}, epoch {epoch}\n"
        f"Members only  (N = {n_members}, Youden threshold)",
        fontsize=15, fontweight="bold", y=0.99,
    )

    ax.set_xlim(-1.12, 1.15)
    ax.set_ylim(-1.15, 1.15)
    plt.tight_layout()
    fig.savefig(
        out_dir / f"{model}_epoch{epoch}_pearl_all_mias_venn.png",
        dpi=200, bbox_inches="tight",
    )
    plt.close(fig)


def plot_pearl_mia_venn_grid(
    df_overlap: pd.DataFrame,
    model: str,
    epoch: int,
    out_dir: Path,
    *,
    pearl: Optional[np.ndarray] = None,
    mia_flags: Optional[Dict[str, np.ndarray]] = None,
    paper_assets_dir: Optional[Path] = None,
) -> None:
    """Large combined figure: pairwise Venns + unified 4-set diagram."""
    sub = df_overlap[(df_overlap["model"] == model) & (df_overlap["epoch"] == epoch)]
    if sub.empty:
        return

    label = MODEL_LABELS.get(model, model)
    n_members = int(sub.iloc[0]["n_members"])
    rows = list(sub.iterrows())
    n = len(rows)

    # ── Panel A: three large pairwise Venns on one canvas ─────────────────────
    fig = plt.figure(figsize=(7.2 * n, 7.5))
    width = 0.94 / n
    for i, (_, row) in enumerate(rows):
        ax = fig.add_axes([0.03 + i * width, 0.12, width * 0.92, 0.78])
        plot_venn2(
            ax,
            pearl_only=int(row["pearl_only"]),
            both=int(row["both"]),
            mia_only=int(row["mia_only"]),
            neither=int(row["neither"]),
            label_pearl="PEARL",
            label_mia=str(row["mia_method"]),
            large=True,
        )
        ax.set_title(
            f"{row['mia_method']}\n"
            f"PEARL={int(row['pearl_flagged'])}  "
            f"MIA={int(row['mia_flagged'])}  "
            f"J={row['jaccard']:.3f}",
            fontsize=FS_SUBPLOT_TITLE, pad=10,
        )

    fig.suptitle(
        f"PEARL vs MIA overlap (members, N={n_members}) — {label}, epoch {epoch}",
        fontsize=FS_SUPTITLE + 2, y=0.98,
    )
    out_pairwise = out_dir / f"{model}_epoch{epoch}_pearl_mia_venn.png"
    fig.savefig(out_pairwise, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if paper_assets_dir is not None:
        paper_assets_dir.mkdir(parents=True, exist_ok=True)
        slug = model.replace("pythia_", "").replace(".", "")
        shutil.copy2(out_pairwise, paper_assets_dir / f"rq7_venn_{slug}.png")

    # ── Panel B: unified 4-set Venn (PEARL + all MIAs together) ───────────────
    if pearl is not None and mia_flags:
        plot_pearl_all_mias_unified_venn(pearl, mia_flags, model, epoch, out_dir)
        if paper_assets_dir is not None:
            unified_src = out_dir / f"{model}_epoch{epoch}_pearl_all_mias_venn.png"
            if unified_src.exists():
                slug = model.replace("pythia_", "").replace(".", "")
                shutil.copy2(unified_src, paper_assets_dir / f"rq7_venn_{slug}_all.png")


def plot_all_models_venn_at_epoch(
    df_overlap: pd.DataFrame, epoch: int, out_dir: Path,
) -> None:
    """Per MIA method: 2×2 grid of Venns for each Pythia model at one epoch."""
    for mia_method in df_overlap["mia_method"].unique():
        sub = df_overlap[
            (df_overlap["epoch"] == epoch) & (df_overlap["mia_method"] == mia_method)
        ]
        models_in = [m for m in MODEL_ORDER if m in sub["model"].values]
        models_in += [m for m in sub["model"].unique() if m not in models_in]
        if not models_in:
            continue

        n = len(models_in)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5 * nrows))
        axes_flat = np.atleast_1d(axes).flatten()

        for ax, model in zip(axes_flat, models_in):
            row = sub[sub["model"] == model]
            if row.empty:
                ax.axis("off")
                continue
            r = row.iloc[0]
            plot_venn2(
                ax,
                pearl_only=int(r["pearl_only"]),
                both=int(r["both"]),
                mia_only=int(r["mia_only"]),
                neither=int(r["neither"]),
                label_pearl="PEARL",
                label_mia=mia_method,
            )
            ax.set_title(MODEL_LABELS.get(model, model), fontsize=FS_SUBPLOT_TITLE)

        for ax in axes_flat[len(models_in):]:
            ax.axis("off")

        slug = mia_method_slug(mia_method)
        plt.suptitle(
            f"PEARL ∩ {mia_method} — members, epoch {epoch} (Youden)",
            fontsize=FS_SUPTITLE, y=1.02,
        )
        plt.tight_layout()
        fig.savefig(
            out_dir / f"cross_model_venn_{slug}_epoch{epoch}.png",
            dpi=150, bbox_inches="tight",
        )
        plt.close(fig)


def build_model_size_scaling_table(
    df_ipa: pd.DataFrame,
    operator: str = "A_mean",
    epochs: Sequence[int] = MODEL_SIZE_EPOCHS,
) -> pd.DataFrame:
    """PEARL/IPA A_mean metrics per model and epoch for scaling plots."""
    if df_ipa.empty:
        return pd.DataFrame()

    sub = df_ipa[df_ipa["operator"] == operator].copy()
    rows = []
    for model in MODEL_ORDER:
        msub = sub[sub["model"] == model]
        if msub.empty:
            continue
        for epoch in epochs:
            esub = msub[msub["epoch"] == epoch]
            if esub.empty:
                continue
            r = esub.iloc[0]
            rows.append({
                "model": model,
                "model_label": MODEL_LABELS.get(model, model),
                "params_M": MODEL_PARAMS_M.get(model),
                "epoch": int(epoch),
                "auc": float(r["auc"]),
                "gamma": float(r["gamma"]),
                "abs_gamma": float(r["abs_gamma"]),
                "mu_S_E": float(r["mu_S_E"]),
                "mu_S_G": float(r["mu_S_G"]),
            })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["epoch", "params_M"])
    return df


def plot_model_size_auc_by_epoch(
    df_scale: pd.DataFrame,
    plot_dir: Path,
    dpi: int = 150,
    epochs: Sequence[int] = MODEL_SIZE_EPOCHS,
) -> None:
    """Line plot: IPA A_mean AUC vs model size at epochs 0, 1, 10."""
    if df_scale.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_MODEL_SCALING)
    epochs_present = [e for e in epochs if e in df_scale["epoch"].values]

    for epoch in epochs_present:
        sub = df_scale[df_scale["epoch"] == epoch].sort_values("params_M")
        if len(sub) < 2:
            continue
        style = EPOCH_LINE_STYLES.get(epoch, {"color": "grey", "marker": "o", "label": f"Epoch {epoch}"})
        ax.plot(
            sub["params_M"], sub["auc"],
            linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color=style["color"], marker=style["marker"], label=style["label"],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['auc']:.3f}",
                (row["params_M"], row["auc"]),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=FS_ANNOT, color=style["color"],
            )

    ax.axhline(0.5, color="k", ls="--", lw=0.8, alpha=0.5, label="Random (0.5)")
    set_axis_labels(
        ax,
        ylabel="PEARL AUC (A_mean, members positive)",
        title="Detection AUC vs model size — Pythia family",
    )
    ax.set_ylim(0.40, 0.88)
    configure_model_size_xaxis(ax)
    plot_legend(ax, loc="lower right")
    plt.tight_layout()
    path = plot_dir / "model_size_auc_epochs_0_1_10.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_model_size_gamma_by_epoch(
    df_scale: pd.DataFrame,
    plot_dir: Path,
    dpi: int = 150,
    epochs: Sequence[int] = MODEL_SIZE_EPOCHS,
) -> None:
    """Line plot: memorization gap |γ| vs model size at epochs 0, 1, 10."""
    if df_scale.empty:
        return

    fig, ax = plt.subplots(figsize=FIG_MODEL_SCALING)
    epochs_present = [e for e in epochs if e in df_scale["epoch"].values]

    for epoch in epochs_present:
        sub = df_scale[df_scale["epoch"] == epoch].sort_values("params_M")
        if len(sub) < 2:
            continue
        style = EPOCH_LINE_STYLES.get(epoch, {"color": "grey", "marker": "o", "label": f"Epoch {epoch}"})
        ax.plot(
            sub["params_M"], sub["abs_gamma"],
            linestyle="-", linewidth=LINE_WIDTH, markersize=MARKER_SIZE,
            color=style["color"], marker=style["marker"], label=style["label"],
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['abs_gamma']:.3f}",
                (row["params_M"], row["abs_gamma"]),
                textcoords="offset points", xytext=(0, 7),
                ha="center", fontsize=FS_ANNOT, color=style["color"],
            )

    set_axis_labels(
        ax,
        ylabel="|γ| = |μ(S_G) − μ(S_E)|",
        title="Memorization gap vs model size — Pythia family",
    )
    configure_model_size_xaxis(ax)
    plot_legend(ax, loc="upper left")
    plt.tight_layout()
    path = plot_dir / "model_size_gamma_epochs_0_1_10.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path}")


def plot_model_size_scaling_evolution(
    df_ipa: pd.DataFrame,
    out_dir: Path,
    plot_dir: Path,
    dpi: int = 150,
    epochs: Sequence[int] = MODEL_SIZE_EPOCHS,
    operator: str = "A_mean",
) -> Optional[pd.DataFrame]:
    """Generate AUC and |γ| vs model size plots for epochs 0, 1, 10."""
    df_scale = build_model_size_scaling_table(df_ipa, operator=operator, epochs=epochs)
    if df_scale.empty or df_scale["model"].nunique() < 2:
        print("  [scaling] insufficient models for size-evolution plots, skip")
        return df_scale

    df_scale.to_csv(out_dir / "model_size_auc_gamma_epochs_0_1_10.csv", index=False)
    print("\n── Model size scaling (epochs 0, 1, 10) ──")
    plot_model_size_auc_by_epoch(df_scale, plot_dir, dpi=dpi, epochs=epochs)
    plot_model_size_gamma_by_epoch(df_scale, plot_dir, dpi=dpi, epochs=epochs)
    return df_scale


# ── Pythia-410M score distribution boxplots ───────────────────────────────────

def collect_score_distributions(
    model: str, epoch: int, run_dir: Path
) -> pd.DataFrame:
    """Long-format per-sample scores for IPA operators, PEARL, and MIA."""
    rows: List[dict] = []
    recs_E = load_records(run_dir / "members" / "records.json")
    recs_G = load_records(run_dir / "non_members" / "records.json")
    if not recs_E or not recs_G:
        return pd.DataFrame()

    for op_name, agg_fn, _ in IPA_OPERATORS:
        for split, recs in (("members", recs_E), ("non_members", recs_G)):
            scores = ipa_scores_by_sample(recs, agg_fn)
            for idx, val in scores.items():
                rows.append({
                    "model": model,
                    "epoch": epoch,
                    "method": f"IPA {op_name}",
                    "method_group": "IPA",
                    "split": split,
                    "sample_idx": idx,
                    "score": val,
                })

    agg_fn_mean = IPA_OPERATORS[0][1]
    for split, recs in (("members", recs_E), ("non_members", recs_G)):
        scores = ipa_scores_by_sample(recs, agg_fn_mean)
        for idx, val in scores.items():
            rows.append({
                "model": model,
                "epoch": epoch,
                "method": "PEARL (A_mean)",
                "method_group": "detection",
                "split": split,
                "sample_idx": idx,
                "score": val,
            })

    for split, fname in (("members", "members"), ("non_members", "non_members")):
        mia_path = run_dir / fname / "mia_summary.csv"
        if not mia_path.exists():
            continue
        df = pd.read_csv(mia_path)
        for col, _, label in MIA_METHODS:
            if col not in df.columns:
                continue
            for _, r in df.iterrows():
                v = r[col]
                if pd.notna(v):
                    rows.append({
                        "model": model,
                        "epoch": epoch,
                        "method": label,
                        "method_group": "detection",
                        "split": split,
                        "sample_idx": int(r["sample_idx"]),
                        "score": float(v),
                    })

    cdd_path = run_dir / "members" / "baseline_results.csv"
    if cdd_path.exists():
        for split, fname in (("members", "members"), ("non_members", "non_members")):
            p = run_dir / fname / "baseline_results.csv"
            if not p.exists():
                continue
            df = pd.read_csv(p)
            if "cdd_score" not in df.columns:
                continue
            for _, r in df.iterrows():
                v = r["cdd_score"]
                if pd.notna(v):
                    rows.append({
                        "model": model,
                        "epoch": epoch,
                        "method": "CDD",
                        "method_group": "detection",
                        "split": split,
                        "sample_idx": int(r["sample_idx"]),
                        "score": float(v),
                    })

    return pd.DataFrame(rows)


def _styled_boxplot(
    ax: plt.Axes,
    scores_E: np.ndarray,
    scores_G: np.ndarray,
    *,
    title: str,
    ylabel: str = "Score",
) -> None:
    if len(scores_E) == 0 or len(scores_G) == 0:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return

    bp = ax.boxplot(
        [scores_E, scores_G],
        tick_labels=["Members", "Non-members"],
        patch_artist=True,
        widths=0.55,
        showfliers=True,
        flierprops={"marker": ".", "markersize": 2.5, "alpha": 0.35},
        medianprops={"color": "black", "linewidth": 1.4},
    )
    bp["boxes"][0].set_facecolor(MEMBER_BOX_COLOR)
    bp["boxes"][1].set_facecolor(NONMEMBER_BOX_COLOR)
    bp["boxes"][0].set_edgecolor("#c0392b")
    bp["boxes"][1].set_edgecolor("#2980b9")
    ax.set_title(title, fontsize=FS_SUBPLOT_TITLE)
    ax.set_ylabel(ylabel, fontsize=FS_TICK)
    style_grid(ax, axis="y")


def plot_pythia_410m_score_boxplots(
    results_root: Path,
    out_dir: Path,
    plot_dir: Path,
    epochs: Sequence[int],
    model: str = BOXPLOT_MODEL_DEFAULT,
    dpi: int = 150,
) -> None:
    """Boxplots of IPA / PEARL / MIA / CDD score distributions (members vs non-members)."""
    model_dir = results_root / model
    if not model_dir.exists():
        print(f"  [boxplot] {model} not found, skip")
        return

    available = discover_epochs(model_dir)
    epochs_to_plot = [e for e in epochs if e in available]
    if not epochs_to_plot:
        epochs_to_plot = [max(available)] if available else []
    if not epochs_to_plot:
        return

    label = MODEL_LABELS.get(model, model)
    print(f"\n── {label} score boxplots  epochs {epochs_to_plot}")

    for epoch in epochs_to_plot:
        run_dir = model_dir / f"run_{epoch}"
        df = collect_score_distributions(model, epoch, run_dir)
        if df.empty:
            continue

        csv_path = out_dir / f"{model}_score_distributions_epoch{epoch}.csv"
        df.to_csv(csv_path, index=False)

        # ── IPA operators (2×3) ─────────────────────────────────────────────
        ipa_methods = [f"IPA {op}" for op, _, _ in IPA_OPERATORS]
        fig_ipa, axes_ipa = plt.subplots(2, 3, figsize=(13, 8))
        for ax, method in zip(axes_ipa.flat, ipa_methods):
            sub = df[df["method"] == method]
            s_E = sub[sub["split"] == "members"]["score"].values
            s_G = sub[sub["split"] == "non_members"]["score"].values
            _styled_boxplot(
                ax, s_E, s_G,
                title=method.replace("IPA ", ""),
                ylabel="Neighborhood score S",
            )
        fig_ipa.suptitle(
            f"{label} — IPA neighbourhood scores (epoch {epoch})",
            fontsize=FS_SUPTITLE, y=1.01,
        )
        plt.tight_layout()
        path_ipa = plot_dir / f"{model}_epoch{epoch}_ipa_scores_boxplot.png"
        fig_ipa.savefig(path_ipa, dpi=dpi, bbox_inches="tight")
        plt.close(fig_ipa)
        print(f"  → {path_ipa}")

        # ── Detection scores: PEARL + MIA (+ CDD if present) ───────────────
        det_methods = ["PEARL (A_mean)"] + [lbl for _, _, lbl in MIA_METHODS]
        if (df["method"] == "CDD").any():
            det_methods.append("CDD")

        n_det = len(det_methods)
        ncols = min(4, n_det)
        nrows = int(np.ceil(n_det / ncols))
        fig_det, axes_det = plt.subplots(nrows, ncols, figsize=(3.8 * ncols, 4.5 * nrows))
        for ax, method in zip(np.atleast_1d(axes_det).flatten(), det_methods):
            sub = df[df["method"] == method]
            s_E = sub[sub["split"] == "members"]["score"].values
            s_G = sub[sub["split"] == "non_members"]["score"].values
            ylab = "Score (raw)"
            if method == "MIA-loss":
                ylab = "MIA loss (↓ = member)"
            elif method in ("MIA-min-K", "MIA-neighborhood"):
                ylab = f"{method} (↑ = member)"
            _styled_boxplot(ax, s_E, s_G, title=method, ylabel=ylab)

        for ax in np.atleast_1d(axes_det).flatten()[len(det_methods):]:
            ax.axis("off")

        fig_det.suptitle(
            f"{label} — detection score distributions (epoch {epoch})",
            fontsize=FS_SUPTITLE, y=1.01,
        )
        plt.tight_layout()
        path_det = plot_dir / f"{model}_epoch{epoch}_detection_scores_boxplot.png"
        fig_det.savefig(path_det, dpi=dpi, bbox_inches="tight")
        plt.close(fig_det)
        print(f"  → {path_det}")

        # ── Combined overview (all methods, shared y only within IPA) ───────
        all_methods = ipa_methods + det_methods
        n_all = len(all_methods)
        ncols = 4
        nrows = int(np.ceil(n_all / ncols))
        fig_all, axes_all = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        for ax, method in zip(np.atleast_1d(axes_all).flatten(), all_methods):
            sub = df[df["method"] == method]
            s_E = sub[sub["split"] == "members"]["score"].values
            s_G = sub[sub["split"] == "non_members"]["score"].values
            short = method.replace("IPA ", "").replace("PEARL ", "PEARL\n")
            _styled_boxplot(ax, s_E, s_G, title=short, ylabel="")
            ax.set_ylabel("")

        for ax in np.atleast_1d(axes_all).flatten()[len(all_methods):]:
            ax.axis("off")

        fig_all.suptitle(
            f"{label} — all scores (epoch {epoch})",
            fontsize=FS_SUPTITLE, y=1.01,
        )
        plt.tight_layout()
        path_all = plot_dir / f"{model}_epoch{epoch}_all_scores_boxplot.png"
        fig_all.savefig(path_all, dpi=dpi, bbox_inches="tight")
        plt.close(fig_all)
        print(f"  → {path_all}")


# ── markdown report ───────────────────────────────────────────────────────────

def write_summary_md(
    out_dir: Path,
    df_ipa: pd.DataFrame,
    df_mia: pd.DataFrame,
    df_cdd: pd.DataFrame,
    df_overlap: pd.DataFrame,
    models: Sequence[str],
) -> None:
    lines = [
        "# IPA / PEARL Results Analysis",
        "",
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Methodology",
        "",
        "Neighborhood aggregation score "
        "`S(x) = A({output_similarity | (x',y') ∈ N(x)})` "
        "with operators from `problem_formulation.tex`.",
        "",
        "- **Memorization gap (paper):** γ = μ(S_G) − μ(S_E)",
        "- **Z-score:** M_score(u) = (μ(S_G) − S(u)) / σ(S_G)",
        "- **IPA rule (τ=1):** S(u) < μ(S_G) − τ·γ when members are more brittle; "
        "S(u) > μ(S_G) − τ·γ when members show higher output stability (empirical PSH)",
        "- **AUC:** oriented so members are the positive class (auto-calibrated)",
        "- **Counts in tables:** Youden-optimal threshold (max TPR − FPR)",
        "",
        "## Models analysed",
        "",
    ]
    for m in models:
        n_ep = df_ipa[df_ipa["model"] == m]["epoch"].nunique() if not df_ipa.empty else 0
        lines.append(f"- **{MODEL_LABELS.get(m, m)}** (`{m}`): {n_ep} epoch(s)")

    if not df_ipa.empty:
        lines.extend(["", "## IPA A_mean — latest epoch per model", ""])
        lines.append(
            "| Model | Epoch | γ | |γ| | AUC | TP | FP | Recall | Precision |"
        )
        lines.append("|-------|-------|---|-----|-----|----|----|--------|-----------|")
        for m in models:
            sub = df_ipa[(df_ipa["model"] == m) & (df_ipa["operator"] == "A_mean")]
            if sub.empty:
                continue
            row = sub.sort_values("epoch").iloc[-1]
            lines.append(
                f"| {row['model_label']} | {int(row['epoch'])} | "
                f"{row['gamma']:.4f} | {row['abs_gamma']:.4f} | {row['auc']:.3f} | "
                f"{int(row['tp'])} | {int(row['fp'])} | {row['recall']:.3f} | "
                f"{row['precision']:.3f} |"
            )

    if not df_mia.empty:
        lines.extend(["", "## MIA baselines (latest epoch)", ""])
        lines.append("| Model | Epoch | Method | AUC | TP | FP |")
        lines.append("|-------|-------|--------|-----|----|----|")
        for m in models:
            sub = df_mia[df_mia["model"] == m]
            if sub.empty:
                continue
            ep = sub["epoch"].max()
            for _, row in sub[sub["epoch"] == ep].iterrows():
                lines.append(
                    f"| {MODEL_LABELS.get(m, m)} | {int(ep)} | {row['method']} | "
                    f"{row['auc']:.3f} | {int(row['tp'])} | {int(row['fp'])} |"
                )

    if not df_overlap.empty:
        lines.extend([
            "",
            "## Flagged instance counts (members, Youden J)",
            "",
            "| Model | Epoch | Method | # flagged |",
            "|-------|-------|--------|-----------|",
        ])
        latest = df_overlap.groupby("model")["epoch"].max()
        for m in models:
            if m not in latest.index:
                continue
            ep = int(latest[m])
            ipa_sub = df_ipa[
                (df_ipa["model"] == m) & (df_ipa["epoch"] == ep)
                & (df_ipa["operator"] == "A_mean")
            ] if not df_ipa.empty else pd.DataFrame()
            if not ipa_sub.empty:
                r0 = ipa_sub.iloc[0]
                lines.append(
                    f"| {MODEL_LABELS.get(m, m)} | {ep} | PEARL | "
                    f"{int(r0['memorized_members_youden'])} |"
                )
            mia_sub = df_mia[(df_mia["model"] == m) & (df_mia["epoch"] == ep)] if not df_mia.empty else pd.DataFrame()
            for _, mr in mia_sub.iterrows():
                lines.append(
                    f"| {MODEL_LABELS.get(m, m)} | {ep} | {mr['method']} | "
                    f"{int(mr['memorized_members'])} |"
                )

        lines.extend([
            "",
            "## PEARL vs MIA overlap on members (Venn regions)",
            "",
            "Columns: **PEARL** / **MIA** = total flagged; **Both** = intersection; "
            "**PEARL only** / **MIA only** = exclusive; **Neither** = not flagged by either.",
            "",
            "| Model | Epoch | MIA | N | PEARL | MIA | Both | PEARL only | MIA only | Neither | Jaccard |",
            "|-------|-------|-----|---|-------|-----|------|------------|----------|---------|---------|",
        ])
        for _, row in df_overlap.iterrows():
            lines.append(
                f"| {MODEL_LABELS.get(row['model'], row['model'])} | "
                f"{int(row['epoch'])} | {row['mia_method']} | {int(row['n_members'])} | "
                f"{int(row['pearl_flagged'])} | {int(row['mia_flagged'])} | "
                f"{int(row['both'])} | {int(row['pearl_only'])} | {int(row['mia_only'])} | "
                f"{int(row['neither'])} | {row['jaccard']:.3f} |"
            )
        lines.append("")
        lines.append(
            "Venn diagrams: `plots/<model>_epoch<E>_pearl_mia_venn.png` (large pairwise), "
            "`plots/<model>_epoch<E>_pearl_all_mias_venn.png` (unified 4-set), and "
            "`plots/cross_model_venn_<mia>_epoch<E>.png`."
        )

    lines.extend([
        "",
        "## Output files",
        "",
        "- `ipa_metrics.csv` — full metrics per model/epoch/operator",
        "- `detection_at_youden.csv` — detection counts (subset of ipa_metrics)",
        "- `mia_metrics.csv`, `cdd_metrics.csv` — baselines when available",
        "- `mia_overlap.csv` — PEARL vs MIA overlap on members (Venn counts)",
        "- `model_epoch_comparison.csv` — cross-model AUC and flagged counts",
        "- `plots/cross_model_auc_epoch*.png` — PEARL + MIA AUC (no CDD)",
        "- `plots/cross_model_flagged_epoch*.png` — flagged member counts",
        "- `plots/*_pearl_mia_venn.png` — large pairwise PEARL vs each MIA",
        "- `plots/*_pearl_all_mias_venn.png` — unified PEARL + all MIAs Venn",
        "- `plots/cross_model_venn_*_epoch*.png` — Venns by MIA method",
        "- `pythia_410m_score_distributions_epoch*.csv` — per-sample scores",
        "- `plots/pythia_410m_epoch*_ipa_scores_boxplot.png` — IPA operator boxplots",
        "- `plots/pythia_410m_epoch*_detection_scores_boxplot.png` — PEARL / MIA / CDD",
        "- `model_size_auc_gamma_epochs_0_1_10.csv` — AUC & γ by model size",
        "- `plots/model_size_auc_epochs_0_1_10.png` — AUC vs size (epochs 0, 1, 10)",
        "- `plots/model_size_gamma_epochs_0_1_10.png` — |γ| vs size (epochs 0, 1, 10)",
        "",
    ])

    (out_dir / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_analysis(
    results_root: Path,
    out_dir: Path,
    models: Optional[List[str]] = None,
    epochs_filter: Optional[List[int]] = None,
    tau: float = 1.0,
    primary_operator: str = "A_mean",
    plot_distributions_for: Optional[List[Tuple[str, int]]] = None,
    boxplot_model: str = BOXPLOT_MODEL_DEFAULT,
    boxplot_epochs: Optional[Sequence[int]] = None,
    skip_boxplots: bool = False,
) -> None:
    plot_dir = out_dir / PLOT_DIR_NAME
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    if models is None:
        models = discover_models(results_root)
    if not models:
        print(f"No models found under {results_root}")
        return

    ipa_rows: List[dict] = []
    mia_rows: List[dict] = []
    cdd_rows: List[dict] = []
    overlap_rows: List[dict] = []
    overlap_flags: Dict[Tuple[str, int], Tuple[np.ndarray, Dict[str, np.ndarray]]] = {}
    instance_rows: List[dict] = []
    comp_rows: List[dict] = []

    dist_targets = set(plot_distributions_for or [])

    print(f"Results: {results_root}")
    print(f"Output:  {out_dir}")
    print(f"Models:  {models}\n")

    for model in models:
        model_dir = results_root / model
        epochs = discover_epochs(model_dir)
        if epochs_filter is not None:
            epochs = [e for e in epochs if e in epochs_filter]
        if not epochs:
            print(f"  [{model}] no epochs, skip")
            continue

        print(f"── {MODEL_LABELS.get(model, model)}  epochs {epochs}")

        for epoch in epochs:
            run_dir = model_dir / f"run_{epoch}"
            recs_E = load_records(run_dir / "members" / "records.json")
            recs_G = load_records(run_dir / "non_members" / "records.json")
            if not recs_E or not recs_G:
                continue

            mia_rows.extend(analyse_mia(model, epoch, run_dir))
            cdd = analyse_cdd(model, epoch, run_dir)
            if cdd:
                cdd_rows.append(cdd)

            ipa_flags_saved = None
            idx_E_saved: List[int] = []

            for op_name, agg_fn, default_lower_S in IPA_OPERATORS:
                m = analyse_ipa_run(
                    model, epoch, recs_E, recs_G,
                    op_name, agg_fn, default_lower_S, tau=tau,
                )
                if m is None:
                    continue
                ipa_rows.append(metrics_to_row(m))

                if op_name == primary_operator:
                    scores_E = ipa_scores_by_sample(recs_E, agg_fn)
                    scores_G = ipa_scores_by_sample(recs_G, agg_fn)
                    s_E, s_G, idx_E, idx_G = align_scores(scores_E, scores_G)
                    sigma_G = float(np.std(s_G))
                    mu_G = float(np.mean(s_G))

                    memb_hi = m.members_higher_S
                    thr, tp, fp, _, _ = youden_threshold(s_E, s_G, memb_hi)
                    det_E = np.array([detection_score(s, memb_hi) for s in s_E]) >= thr
                    ipa_flags_saved = det_E
                    idx_E_saved = idx_E

                    if (model, epoch) in dist_targets or (
                        not dist_targets and epoch == max(epochs)
                    ):
                        plot_score_distributions(
                            s_E, s_G, model, epoch, op_name,
                            m.mu_E, m.mu_G, m.gamma, tau, plot_dir,
                        )

                    for i, s in zip(idx_E, s_E):
                        instance_rows.append({
                            "model": model,
                            "epoch": epoch,
                            "split": "members",
                            "sample_idx": i,
                            "operator": op_name,
                            "S": round(float(s), 6),
                            "M_score": round(memorization_score_z(float(s), mu_G, sigma_G), 4),
                            "flagged_youden": bool(
                                detection_score(float(s), memb_hi) >= thr
                            ),
                            "flagged_ipa_rule": bool(
                                (float(s) > mu_G - tau * m.gamma)
                                if memb_hi
                                else (float(s) < mu_G - tau * m.gamma)
                            ),
                            "members_higher_S": memb_hi,
                        })

                    comp = {
                        "model": model,
                        "model_label": MODEL_LABELS.get(model, model),
                        "params_M": MODEL_PARAMS_M.get(model),
                        "epoch": epoch,
                        "ipa_auc": round(m.auc, 4),
                        "ipa_gamma": round(m.gamma, 6),
                        "ipa_abs_gamma": round(abs(m.gamma), 6),
                        "ipa_tp": m.tp,
                        "ipa_fp": m.fp,
                        "ipa_flagged_members": m.tp,
                        "ipa_flagged_non_members": m.fp,
                    }
                    if cdd:
                        comp["cdd_auc"] = cdd["auc"]
                    sub_mia = [
                        r for r in mia_rows
                        if r["model"] == model and r["epoch"] == epoch
                    ]
                    for r in sub_mia:
                        slug = mia_method_slug(r["method"])
                        comp[f"{slug}_auc"] = r["auc"]
                        comp[f"{slug}_flagged_members"] = r["memorized_members"]
                        comp[f"{slug}_flagged_non_members"] = r["memorized_non_members"]
                    comp_rows.append(comp)

            if ipa_flags_saved is not None and idx_E_saved:
                overlap_rows.extend(
                    compute_mia_overlap(
                        model, epoch, run_dir,
                        ipa_flags_saved, idx_E_saved,
                    )
                )
                gathered = gather_pearl_mia_member_flags(
                    run_dir, ipa_flags_saved, idx_E_saved,
                )
                if gathered is not None:
                    pearl_f, mia_f, _ = gathered
                    overlap_flags[(model, epoch)] = (pearl_f, mia_f)

            if epoch == max(epochs) or (model, epoch) in dist_targets:
                plot_similarity_curves(run_dir, model, epoch, plot_dir)

        df_model = pd.DataFrame([r for r in ipa_rows if r["model"] == model])
        if not df_model.empty:
            plot_auc_gamma_epochs(df_model, model, plot_dir)
            df_det = df_model  # same columns
            plot_flagged_counts(df_det, model, primary_operator, plot_dir)

    df_ipa = pd.DataFrame(ipa_rows)
    df_mia = pd.DataFrame(mia_rows)
    df_cdd = pd.DataFrame(cdd_rows)
    df_overlap = pd.DataFrame(overlap_rows)
    df_inst = pd.DataFrame(instance_rows)
    df_comp = pd.DataFrame(comp_rows)

    if not df_ipa.empty:
        df_ipa.to_csv(out_dir / "ipa_metrics.csv", index=False)
        cols_det = [
            "model", "model_label", "epoch", "operator", "auc", "gamma",
            "abs_gamma", "tp", "fp", "fn", "tn", "recall", "precision",
            "memorized_members_youden", "memorized_non_members_youden",
            "flagged_members_ipa_rule", "flagged_non_members_ipa_rule",
        ]
        df_ipa[cols_det].to_csv(out_dir / "detection_at_youden.csv", index=False)

    if not df_mia.empty:
        df_mia.to_csv(out_dir / "mia_metrics.csv", index=False)
    if not df_cdd.empty:
        df_cdd.to_csv(out_dir / "cdd_metrics.csv", index=False)
    if not df_overlap.empty:
        df_overlap.to_csv(out_dir / "mia_overlap.csv", index=False)
    if not df_inst.empty:
        df_inst.to_csv(out_dir / "instance_scores.csv", index=False)
    if not df_comp.empty:
        df_comp.to_csv(out_dir / "model_epoch_comparison.csv", index=False)
        for ep in sorted(df_comp["epoch"].unique()):
            if ep in (1, 10) or ep == df_comp["epoch"].max():
                plot_cross_model_bars(df_comp, int(ep), plot_dir)
                plot_cross_model_flagged_bars(df_comp, int(ep), plot_dir)

    if not df_ipa.empty:
        plot_model_size_scaling_evolution(df_ipa, out_dir, plot_dir)

    if not df_overlap.empty:
        paper_assets = _EVAL_DIR.parent / "paper" / "_assets"
        venn_epochs = sorted(df_overlap["epoch"].unique())
        key_epochs = [e for e in (1, 10) if e in venn_epochs]
        if not key_epochs:
            key_epochs = [int(max(venn_epochs))]
        for ep in key_epochs:
            plot_all_models_venn_at_epoch(df_overlap, int(ep), plot_dir)
            for model in models:
                if model not in df_overlap["model"].values:
                    continue
                flags = overlap_flags.get((model, int(ep)))
                plot_pearl_mia_venn_grid(
                    df_overlap, model, int(ep), plot_dir,
                    pearl=flags[0] if flags else None,
                    mia_flags=flags[1] if flags else None,
                    paper_assets_dir=paper_assets,
                )

    write_summary_md(out_dir, df_ipa, df_mia, df_cdd, df_overlap, models)

    if not skip_boxplots:
        if boxplot_epochs is not None:
            bp_epochs = list(boxplot_epochs)
        elif epochs_filter is not None:
            bp_epochs = list(epochs_filter)
        else:
            bp_epochs = list(BOXPLOT_EPOCHS_DEFAULT)
        plot_pythia_410m_score_boxplots(
            results_root, out_dir, plot_dir,
            epochs=bp_epochs,
            model=boxplot_model,
        )

    print(f"\nWrote reports to {out_dir}")
    if not df_ipa.empty:
        print("\nIPA A_mean (last epoch per model):")
        for m in models:
            sub = df_ipa[(df_ipa["model"] == m) & (df_ipa["operator"] == "A_mean")]
            if sub.empty:
                continue
            row = sub.sort_values("epoch").iloc[-1]
            print(
                f"  {row['model_label']:12s} ep={int(row['epoch']):2d}  "
                f"AUC={row['auc']:.3f}  γ={row['gamma']:+.4f}  "
                f"TP={int(row['tp'])} FP={int(row['fp'])}"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IPA/PEARL analysis over src_v3/results",
    )
    p.add_argument(
        "--results-dir", type=Path, default=DEFAULT_RESULTS,
        help="Root directory containing per-model run folders",
    )
    p.add_argument(
        "--out-dir", type=Path, default=DEFAULT_OUT,
        help="Directory for CSV reports and plots",
    )
    p.add_argument(
        "--models", nargs="+", default=None,
        help="Model tags (default: auto-detect)",
    )
    p.add_argument(
        "--epochs", nargs="+", type=int, default=None,
        help="Restrict to these epoch indices",
    )
    p.add_argument(
        "--tau", type=float, default=1.0,
        help="IPA detection sensitivity τ (default 1.0)",
    )
    p.add_argument(
        "--operator", default="A_mean",
        help="Primary operator for overlap / flag plots",
    )
    p.add_argument(
        "--plot-dist", nargs=2, metavar=("MODEL", "EPOCH"), action="append",
        help="Extra score-distribution plot, e.g. pythia_410m 10",
    )
    p.add_argument(
        "--boxplot-model", default=BOXPLOT_MODEL_DEFAULT,
        help="Model tag for score boxplots (default: pythia_410m)",
    )
    p.add_argument(
        "--boxplot-epochs", nargs="+", type=int, default=None,
        help="Epochs for boxplots (default: 1 and 10)",
    )
    p.add_argument(
        "--no-boxplots", action="store_true",
        help="Skip Pythia-410M score distribution boxplots",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dist = None
    if args.plot_dist:
        dist = [(m, int(e)) for m, e in args.plot_dist]
    run_analysis(
        results_root=args.results_dir.resolve(),
        out_dir=args.out_dir.resolve(),
        models=args.models,
        epochs_filter=args.epochs,
        tau=args.tau,
        primary_operator=args.operator,
        plot_distributions_for=dist,
        boxplot_model=args.boxplot_model,
        boxplot_epochs=args.boxplot_epochs,
        skip_boxplots=args.no_boxplots,
    )


if __name__ == "__main__":
    main()
