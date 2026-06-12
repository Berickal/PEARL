"""
N-gram Memorization Analysis — Carlini et al. (arXiv:2005.14165)
================================================================
For each epoch checkpoint, checks whether the model's generated output
(`original_output` from records.json) contains any verbatim n-gram sequence
that appears in the reference continuation (`reference` in the data file).

Definition (Carlini et al.):
  A sample is "memorized" if the model's generated output shares at least one
  consecutive n-word sequence with the ground-truth training text.

Scores computed per sample:
  - ngram_{n}_score    : fraction of output n-grams that appear in the reference
                         (0 if the output has fewer than n words)
  - ngram_{n}_any      : 1 if any output n-gram matches the reference, else 0

For each n in {5, 7, 13}:
  - coverage           : fraction of samples that have >= n output words
  - μ(S_E), μ(S_G)     : mean score on members / non-members
  - γ = μ(S_E) − μ(S_G): memorization gap
  - AUC                : ROC-AUC

Outputs:
  analysis/results/ngram_per_epoch.csv
  analysis/results/ngram_sample_scores/run_{epoch}_{split}.csv  (per-sample)
  analysis/results/plots/ngram_auc_per_epoch.png
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR   = Path(__file__).resolve().parent.parent
RESULTS   = SRC_DIR / "results" / "pythia_70m"
DATA_DIR  = SRC_DIR / "data"
OUT_DIR   = Path(__file__).resolve().parent / "results"
PLOT_DIR  = OUT_DIR / "plots"
SCORE_DIR = OUT_DIR / "ngram_sample_scores"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
SCORE_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILES = {
    "members":     DATA_DIR / "mimir_ngram_13_0.8_members.json",
    "non_members": DATA_DIR / "mimir_ngram_13_0.8_non_members.json",
}

EPOCHS = list(range(11))
N_VALUES = [5, 7, 13]


# ── n-gram helpers ────────────────────────────────────────────────────────────

def word_ngrams(text: str, n: int) -> set:
    """Return the set of word-level n-grams (as tuples) from text."""
    words = text.split()
    if len(words) < n:
        return set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def ngram_score(output: str, reference: str, n: int) -> tuple[float, int]:
    """
    Returns (fraction_score, any_match).

    fraction_score : |output_ngrams ∩ ref_ngrams| / |output_ngrams|
                     0.0 if output has fewer than n words.
    any_match      : 1 if at least one output n-gram appears in the reference.
    """
    out_grams = word_ngrams(output, n)
    if not out_grams:
        return 0.0, 0
    ref_grams = word_ngrams(reference, n)
    matches = len(out_grams & ref_grams)
    return matches / len(out_grams), int(matches > 0)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_references(split: str) -> dict[int, str]:
    """Return {sample_idx: reference_text} from the raw data file."""
    with open(DATA_FILES[split], encoding="utf-8") as f:
        data = json.load(f)
    return {i: d.get("reference", "") for i, d in enumerate(data)}


def load_outputs(epoch: int, split: str) -> dict[int, str]:
    """
    Return {sample_idx: original_output} from records.json.
    Takes the first occurrence per sample (original_output is the same for
    all perturbation records of the same sample).
    """
    path = RESULTS / f"run_{epoch}" / split / "records.json"
    if not path.exists():
        print(f"  [MISSING] {path}", file=sys.stderr)
        return {}
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    outputs: dict[int, str] = {}
    for r in records:
        idx = r["sample_idx"]
        if idx not in outputs:
            outputs[idx] = r.get("original_output", "") or ""
    return outputs


# ── per-sample scoring ────────────────────────────────────────────────────────

def score_split(epoch: int, split: str) -> pd.DataFrame | None:
    refs    = load_references(split)
    outputs = load_outputs(epoch, split)
    if not outputs:
        return None

    rows = []
    for idx in sorted(outputs):
        ref = refs.get(idx, "")
        out = outputs[idx]
        row: dict = {"sample_idx": idx, "output_words": len(out.split())}
        for n in N_VALUES:
            score, any_m = ngram_score(out, ref, n)
            row[f"ngram_{n}_score"] = round(score, 6)
            row[f"ngram_{n}_any"]   = any_m
        rows.append(row)

    return pd.DataFrame(rows)


# ── AUC helper ────────────────────────────────────────────────────────────────

def compute_auc(scores_E: list, scores_G: list) -> float:
    """Higher score → more memorized (same direction as Carlini: overlap = memorized)."""
    y_true  = [1] * len(scores_E) + [0] * len(scores_G)
    y_score = scores_E + scores_G
    if len(set(y_true)) < 2:
        return float("nan")
    return roc_auc_score(y_true, y_score)


# ── main loop ─────────────────────────────────────────────────────────────────

summary_rows = []

for epoch in EPOCHS:
    print(f"Epoch {epoch} …")

    dfs: dict[str, pd.DataFrame] = {}
    for split in ("members", "non_members"):
        df = score_split(epoch, split)
        if df is None:
            break
        # save per-sample scores
        df.to_csv(SCORE_DIR / f"run_{epoch}_{split}.csv", index=False)
        dfs[split] = df
    else:
        df_E = dfs["members"]
        df_G = dfs["non_members"]

        row: dict = {"epoch": epoch}
        for n in N_VALUES:
            col = f"ngram_{n}_score"
            s_E = df_E[col].tolist()
            s_G = df_G[col].tolist()
            mu_E = float(np.mean(s_E))
            mu_G = float(np.mean(s_G))
            gamma = mu_E - mu_G          # positive = members score higher (memorized)
            auc   = compute_auc(s_E, s_G)

            cov_E = (df_E["output_words"] >= n).mean()
            cov_G = (df_G["output_words"] >= n).mean()

            print(f"  n={n:2d}  μ_E={mu_E:.4f}  μ_G={mu_G:.4f}  "
                  f"γ={gamma:+.4f}  AUC={auc:.4f}  "
                  f"cov_E={cov_E:.2%}  cov_G={cov_G:.2%}")

            row[f"n{n}_mu_E"]    = round(mu_E, 6)
            row[f"n{n}_mu_G"]    = round(mu_G, 6)
            row[f"n{n}_gamma"]   = round(gamma, 6)
            row[f"n{n}_auc"]     = round(auc, 4)
            row[f"n{n}_cov_E"]   = round(float(cov_E), 4)
            row[f"n{n}_cov_G"]   = round(float(cov_G), 4)

            # memorized counts (any-match)
            any_col = f"ngram_{n}_any"
            row[f"n{n}_mem_E"] = int(df_E[any_col].sum())
            row[f"n{n}_mem_G"] = int(df_G[any_col].sum())

        summary_rows.append(row)
        continue
    print(f"  skipping epoch {epoch}: missing data")

# ── save summary ──────────────────────────────────────────────────────────────

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUT_DIR / "ngram_per_epoch.csv", index=False)
print(f"\nSaved → {OUT_DIR / 'ngram_per_epoch.csv'}")

# ── print pivot tables ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("AUC per epoch × n")
print("=" * 70)
auc_cols = {f"n{n}_auc": f"n={n}" for n in N_VALUES}
piv = df_summary[["epoch"] + list(auc_cols)].rename(columns=auc_cols)
piv = piv.set_index("epoch")
print(piv.to_string())

print("\n" + "=" * 70)
print("Memorization gap γ = μ(S_E) − μ(S_G) per epoch × n")
print("=" * 70)
g_cols = {f"n{n}_gamma": f"n={n}" for n in N_VALUES}
piv2 = df_summary[["epoch"] + list(g_cols)].rename(columns=g_cols).set_index("epoch")
print(piv2.to_string())

print("\n" + "=" * 70)
print("Memorized instance counts (any match) per epoch × n")
print("=" * 70)
header = f"{'Epoch':>5}  " + "  ".join(
    f"n={n} mem/mem  n={n} nonm/nonm" for n in N_VALUES
)
print(header)
for _, r in df_summary.iterrows():
    parts = "  ".join(
        f"{int(r[f'n{n}_mem_E']):4d}/{int(r[f'n{n}_cov_E']*1000):4d}  "
        f"{int(r[f'n{n}_mem_G']):4d}/{int(r[f'n{n}_cov_G']*1000):4d}"
        for n in N_VALUES
    )
    print(f"{int(r['epoch']):>5}  {parts}")

# ── plots ─────────────────────────────────────────────────────────────────────

COLORS = {5: "#1f77b4", 7: "#d62728", 13: "#2ca02c"}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# AUC
ax = axes[0]
for n in N_VALUES:
    sub = df_summary[["epoch", f"n{n}_auc"]].dropna()
    ax.plot(sub["epoch"], sub[f"n{n}_auc"], marker="o",
            label=f"n={n}", color=COLORS[n])
ax.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.5,
           label="Random (0.5)")
ax.set_xlabel("Epoch")
ax.set_ylabel("AUC")
ax.set_title("N-gram Memorization AUC across epochs")
ax.legend(fontsize=9)
ax.set_xticks(EPOCHS)
ax.set_ylim(0.4, 1.0)
ax.grid(True, alpha=0.3)

# γ
ax = axes[1]
for n in N_VALUES:
    sub = df_summary[["epoch", f"n{n}_gamma"]].dropna()
    ax.plot(sub["epoch"], sub[f"n{n}_gamma"], marker="o",
            label=f"n={n}", color=COLORS[n])
ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("γ = μ(S_E) − μ(S_G)")
ax.set_title("N-gram Memorization Gap γ across epochs")
ax.legend(fontsize=9)
ax.set_xticks(EPOCHS)
ax.grid(True, alpha=0.3)

plt.suptitle("N-gram Analysis — Pythia-70m (Carlini et al. 2005.14165)", y=1.01)
plt.tight_layout()
plt.savefig(PLOT_DIR / "ngram_auc_per_epoch.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"\nPlot saved → {PLOT_DIR / 'ngram_auc_per_epoch.png'}")
print("Done.")
