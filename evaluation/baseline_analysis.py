"""
Baseline Analysis — CDD, CoDeC, ACR across epochs.

Reads baseline_results.csv files produced by run_baseline_sweep.py and
computes AUC (members vs non-members) per method per epoch.

Outputs:
  analysis/results/baseline_auc_per_epoch.csv
  analysis/results/plots/baseline_auc_per_epoch.png
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

SRC_DIR  = Path(__file__).resolve().parent.parent
RESULTS  = SRC_DIR / "results" / "pythia_70m"
OUT_DIR  = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = list(range(11))

# (column, higher_score_means_memorized)
METHODS = {
    "cdd_score":    True,   # higher peakedness → more memorized
    "codec_score":  True,   # 1.0 = contaminated → higher is memorized
    "acr_ratio":    False,  # lower ratio → more memorized
}

COLORS = {
    "cdd_score":   "#e377c2",
    "codec_score": "#17becf",
    "acr_ratio":   "#bcbd22",
}

LABELS = {
    "cdd_score":   "CDD (peakedness)",
    "codec_score": "CoDeC",
    "acr_ratio":   "ACR (compression ratio)",
}


def compute_auc(scores_E, scores_G, higher_is_memorized: bool) -> float:
    y_true  = [1] * len(scores_E) + [0] * len(scores_G)
    raw     = scores_E + scores_G
    y_score = raw if higher_is_memorized else [-s for s in raw]
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


rows = []

for epoch in EPOCHS:
    m_path = RESULTS / f"run_{epoch}" / "members"     / "baseline_results.csv"
    g_path = RESULTS / f"run_{epoch}" / "non_members" / "baseline_results.csv"

    if not m_path.exists() or not g_path.exists():
        print(f"Epoch {epoch}: missing baseline_results.csv, skipping")
        continue

    df_E = pd.read_csv(m_path)
    df_G = pd.read_csv(g_path)

    print(f"Epoch {epoch}:")
    for col, higher in METHODS.items():
        if col not in df_E.columns or col not in df_G.columns:
            continue
        s_E = df_E[col].dropna().tolist()
        s_G = df_G[col].dropna().tolist()
        if not s_E or not s_G:
            continue
        auc = compute_auc(s_E, s_G, higher)
        print(f"  {col:20s}  AUC={auc:.4f}  "
              f"(n_E={len(s_E)}, n_G={len(s_G)})")
        rows.append({"epoch": epoch, "method": col, "auc": round(auc, 4)})

if not rows:
    print("No baseline_results.csv files found. Run run_baseline_sweep.py first.")
    sys.exit(0)

df_auc = pd.DataFrame(rows)
df_auc.to_csv(OUT_DIR / "baseline_auc_per_epoch.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'baseline_auc_per_epoch.csv'}")

print("\n" + "=" * 60)
print("AUC per epoch × method")
print("=" * 60)
piv = df_auc.pivot(index="epoch", columns="method", values="auc")
print(piv.to_string())

# ── plot ──────────────────────────────────────────────────────────────────────
available_methods = df_auc["method"].unique()

fig, ax = plt.subplots(figsize=(10, 5))
for col in available_methods:
    sub = df_auc[df_auc["method"] == col]
    ax.plot(sub["epoch"], sub["auc"], marker="o",
            label=LABELS.get(col, col), color=COLORS.get(col, None))

ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
           label="Random (0.5)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("Baseline Detection AUC across epochs — Pythia-70m")
ax.legend(fontsize=9)
ax.set_xticks(EPOCHS)
ax.set_ylim(0.3, 1.0)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PLOT_DIR / "baseline_auc_per_epoch.png", dpi=150)
plt.close()
print(f"Plot saved → {PLOT_DIR / 'baseline_auc_per_epoch.png'}")
print("Done.")
