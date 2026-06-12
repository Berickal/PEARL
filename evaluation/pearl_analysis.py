"""
PEARL Analysis — RQ5: Effect of Repeated Exposure
==================================================
For each epoch checkpoint (run_0 … run_10), computes:

  IPA scores per sample using 6 aggregation operators over output_similarity:
    - A_mean     : arithmetic mean
    - A_min      : minimum (worst-case robustness)
    - A_neg_var  : negative variance (consistency)
    - A_q10      : 10th-percentile
    - A_q25      : 25th-percentile
    - A_median   : median (50th-percentile)

  For each operator:
    - μ(S_E)   : mean IPA score on members
    - μ(S_G)   : mean IPA score on non-members
    - γ = μ(S_G) − μ(S_E)   : memorization gap (paper Def. 2)
    - AUC  : ROC-AUC

  MIA baselines (per-sample scores from mia_summary.csv):
    - mia_loss         : lower → more memorized  → negate for AUC
    - mia_min_k        : higher → more memorized
    - mia_neighborhood : higher → more memorized

Outputs:
  analysis/results/gamma_per_epoch.csv
  analysis/results/auc_per_epoch.csv
  analysis/results/plots/gamma_*.png
  analysis/results/plots/auc_*.png
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR    = Path(__file__).resolve().parent.parent
RESULTS    = SRC_DIR / "results" / "pythia_70m"
OUT_DIR    = Path(__file__).resolve().parent / "results"
PLOT_DIR   = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = list(range(11))   # 0 … 10

# ── aggregation operators ─────────────────────────────────────────────────────
def agg_mean(vals):    return float(np.mean(vals))
def agg_min(vals):     return float(np.min(vals))
def agg_neg_var(vals): return float(-np.var(vals))
def agg_q10(vals):     return float(np.quantile(vals, 0.10))
def agg_q25(vals):     return float(np.quantile(vals, 0.25))
def agg_median(vals):  return float(np.median(vals))

OPERATORS = {
    "A_mean":    agg_mean,
    "A_min":     agg_min,
    "A_neg_var": agg_neg_var,
    "A_q10":     agg_q10,
    "A_q25":     agg_q25,
    "A_median":  agg_median,
}

# ── helpers ───────────────────────────────────────────────────────────────────

def load_records(epoch: int, split: str) -> list:
    path = RESULTS / f"run_{epoch}" / split / "records.json"
    if not path.exists():
        print(f"  [MISSING] {path}", file=sys.stderr)
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mia(epoch: int, split: str) -> pd.DataFrame:
    path = RESULTS / f"run_{epoch}" / split / "mia_summary.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def compute_ipa_scores(records: list, agg_fn) -> dict:
    """
    Group output_similarity values by sample_idx, apply aggregation,
    return {sample_idx: score}.
    """
    by_sample = defaultdict(list)
    for r in records:
        sim = r.get("output_similarity")
        if sim is not None:
            by_sample[r["sample_idx"]].append(sim)
    return {idx: agg_fn(vals) for idx, vals in by_sample.items() if vals}


def auc_members_vs_nonmembers(scores_E: list, scores_G: list,
                               higher_is_memorized: bool = False) -> float:
    y_true  = [1] * len(scores_E) + [0] * len(scores_G)
    raw     = scores_E + scores_G
    y_score = raw if higher_is_memorized else [-s for s in raw]
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


# ── main loop ─────────────────────────────────────────────────────────────────

gamma_rows = []   # one row per (epoch, operator)
auc_rows   = []   # one row per (epoch, method)

for epoch in EPOCHS:
    print(f"Epoch {epoch} …")

    recs_E = load_records(epoch, "members")
    recs_G = load_records(epoch, "non_members")

    if not recs_E or not recs_G:
        print(f"  skipping epoch {epoch}: missing data")
        continue

    mia_E = load_mia(epoch, "members")
    mia_G = load_mia(epoch, "non_members")

    # ── IPA operators ─────────────────────────────────────────────────────────
    for op_name, op_fn in OPERATORS.items():
        ipa_E = compute_ipa_scores(recs_E, op_fn)
        ipa_G = compute_ipa_scores(recs_G, op_fn)

        scores_E = list(ipa_E.values())
        scores_G = list(ipa_G.values())

        mu_E = float(np.mean(scores_E)) if scores_E else float("nan")
        mu_G = float(np.mean(scores_G)) if scores_G else float("nan")
        gamma = mu_G - mu_E   # paper Def. 2: γ = A(S_G) − A(S_E)

        auc = auc_members_vs_nonmembers(scores_E, scores_G,
                                         higher_is_memorized=False)

        gamma_rows.append({
            "epoch": epoch, "operator": op_name,
            "mu_E": round(mu_E, 6), "mu_G": round(mu_G, 6),
            "gamma": round(gamma, 6),
        })
        auc_rows.append({
            "epoch": epoch, "method": op_name,
            "auc": round(auc, 4),
        })
        print(f"  {op_name:12s}  μ_E={mu_E:.4f}  μ_G={mu_G:.4f}  "
              f"γ={gamma:+.4f}  AUC={auc:.4f}")

    # ── MIA baselines ─────────────────────────────────────────────────────────
    mia_configs = [
        ("mia_loss",         False),   # lower loss → more memorized → negate
        ("mia_min_k",        True),    # higher → more memorized
        ("mia_neighborhood", True),    # higher → more memorized
    ]
    for col, higher in mia_configs:
        if col not in mia_E.columns or col not in mia_G.columns:
            continue
        s_E = mia_E[col].dropna().tolist()
        s_G = mia_G[col].dropna().tolist()
        if not s_E or not s_G:
            continue
        auc = auc_members_vs_nonmembers(s_E, s_G, higher_is_memorized=higher)
        auc_rows.append({
            "epoch": epoch, "method": col,
            "auc": round(auc, 4),
        })
        print(f"  {col:20s}  AUC={auc:.4f}")

# ── save CSVs ─────────────────────────────────────────────────────────────────

df_gamma = pd.DataFrame(gamma_rows)
df_auc   = pd.DataFrame(auc_rows)

df_gamma.to_csv(OUT_DIR / "gamma_per_epoch.csv", index=False)
df_auc.to_csv(OUT_DIR / "auc_per_epoch.csv",   index=False)

print("\nCSVs saved.")

# ── pivot tables for display ──────────────────────────────────────────────────

print("\n" + "=" * 70)
print("MEMORIZATION GAP γ = μ(S_G) − μ(S_E)  per epoch × operator")
print("=" * 70)
piv_gamma = df_gamma.pivot(index="epoch", columns="operator", values="gamma")
print(piv_gamma.to_string())

print("\n" + "=" * 70)
print("AUC (members vs non-members)  per epoch × method")
print("=" * 70)
piv_auc = df_auc.pivot(index="epoch", columns="method", values="auc")
print(piv_auc.to_string())

# ── plots ─────────────────────────────────────────────────────────────────────

OP_COLORS = {
    "A_mean":    "#1f77b4",
    "A_min":     "#d62728",
    "A_neg_var": "#2ca02c",
    "A_q10":     "#ff7f0e",
    "A_q25":     "#9467bd",
    "A_median":  "#8c564b",
}

MIA_COLORS = {
    "mia_loss":         "#e377c2",
    "mia_min_k":        "#17becf",
    "mia_neighborhood": "#bcbd22",
}

# — γ per epoch ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for op in OPERATORS:
    sub = df_gamma[df_gamma["operator"] == op]
    ax.plot(sub["epoch"], sub["gamma"], marker="o",
            label=op, color=OP_COLORS[op])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Epoch (fine-tuning steps)")
ax.set_ylabel("γ = μ(S_G) − μ(S_E)")
ax.set_title("Memorization Gap γ across epochs — Pythia-70m (RQ5)")
ax.legend(loc="upper left", fontsize=8)
ax.set_xticks(EPOCHS)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "gamma_per_epoch.png", dpi=150)
plt.close()

# — μ_E vs μ_G per epoch (one subplot per operator) --------------------------
fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
for ax, op in zip(axes.flat, OPERATORS):
    sub = df_gamma[df_gamma["operator"] == op]
    ax.plot(sub["epoch"], sub["mu_E"], marker="o", label="Members (μ_E)",
            color="#d62728")
    ax.plot(sub["epoch"], sub["mu_G"], marker="s", label="Non-members (μ_G)",
            color="#1f77b4")
    ax.fill_between(sub["epoch"], sub["mu_E"], sub["mu_G"],
                    alpha=0.15, color="grey")
    ax.set_title(op, fontsize=10)
    ax.set_xticks(EPOCHS)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
fig.suptitle("μ(S_E) vs μ(S_G) per aggregation operator — Pythia-70m", y=1.01)
fig.supxlabel("Epoch")
fig.supylabel("IPA score")
plt.tight_layout()
plt.savefig(PLOT_DIR / "mu_E_vs_mu_G_per_operator.png", dpi=150,
            bbox_inches="tight")
plt.close()

# — AUC per epoch (IPA operators) ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for op in OPERATORS:
    sub = df_auc[df_auc["method"] == op]
    ax.plot(sub["epoch"], sub["auc"], marker="o",
            label=op, color=OP_COLORS[op])
ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
           label="Random (0.5)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("IPA Detection AUC across epochs — Pythia-70m")
ax.legend(loc="upper left", fontsize=8)
ax.set_xticks(EPOCHS)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "auc_ipa_per_epoch.png", dpi=150)
plt.close()

# — AUC per epoch (MIA baselines) ---------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
for col, color in MIA_COLORS.items():
    sub = df_auc[df_auc["method"] == col]
    if sub.empty:
        continue
    ax.plot(sub["epoch"], sub["auc"], marker="s",
            label=col, color=color, linestyle="--")
ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5,
           label="Random (0.5)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("MIA Baseline AUC across epochs — Pythia-70m")
ax.legend(fontsize=8)
ax.set_xticks(EPOCHS)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "auc_mia_per_epoch.png", dpi=150)
plt.close()

# — Combined AUC (IPA best + MIA) ---------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
for op in OPERATORS:
    sub = df_auc[df_auc["method"] == op]
    ax.plot(sub["epoch"], sub["auc"], marker="o",
            color=OP_COLORS[op], label=op, alpha=0.8)
for col, color in MIA_COLORS.items():
    sub = df_auc[df_auc["method"] == col]
    if sub.empty:
        continue
    ax.plot(sub["epoch"], sub["auc"], marker="s",
            color=color, label=col, linestyle="--", linewidth=1.5)
ax.axhline(0.5, color="black", linewidth=0.8, linestyle=":", alpha=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("IPA vs MIA Detection AUC — Pythia-70m (RQ5)")
ax.legend(loc="upper left", fontsize=7, ncol=2)
ax.set_xticks(EPOCHS)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "auc_combined_per_epoch.png", dpi=150)
plt.close()

print("\nPlots saved to:", PLOT_DIR)
print("Done.")
