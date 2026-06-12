"""
Multi-model analysis: IPA/PEARL, CDD, MIA across model sizes and epochs.

Computes per-epoch metrics for every model that has results under
results/<model_tag>/run_<epoch>/:
  - IPA scores (6 aggregation operators) from records.json
  - CDD peakedness from baseline_results.csv  (if present)
  - MIA scores (loss, min_k, neighborhood) from mia_summary.csv (if present)

Outputs
-------
analysis/results/<model_tag>_ipa_per_epoch.csv
analysis/results/<model_tag>_cdd_per_epoch.csv   (when CDD data exists)
analysis/results/<model_tag>_mia_per_epoch.csv   (when MIA data exists)
analysis/results/model_size_comparison.csv        (cross-model, available epochs)
analysis/results/plots/<model_tag>_*.png
analysis/results/plots/model_size_comparison*.png

Usage
-----
python analysis/multi_model_analysis.py                    # all models auto-detected
python analysis/multi_model_analysis.py --models pythia_70m pythia_410m
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ── paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).resolve().parent.parent
RESULTS  = SRC_DIR / "results"
OUT_DIR  = Path(__file__).resolve().parent / "results"
PLOT_DIR = OUT_DIR / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# model display order and labels
MODEL_ORDER  = ["pythia_70m", "pythia_410m", "pythia_1.4b", "pythia_2.8b"]
MODEL_LABELS = {
    "pythia_70m":  "Pythia-70M",
    "pythia_410m": "Pythia-410M",
    "pythia_1.4b": "Pythia-1.4B",
    "pythia_2.8b": "Pythia-2.8B",
}
MODEL_PARAMS = {"pythia_70m": 70, "pythia_410m": 410,
                "pythia_1.4b": 1400, "pythia_2.8b": 2800}
MODEL_COLORS = {
    "pythia_70m":  "#1f77b4",
    "pythia_410m": "#d62728",
    "pythia_1.4b": "#2ca02c",
    "pythia_2.8b": "#ff7f0e",
}

# IPA operators: (name, inverted_psh) — inverted=True means members have HIGHER
# output similarity, so we flip the AUC direction (1 - raw_auc)
IPA_OPS = [
    ("A_mean",    True),
    ("A_min",     True),
    ("A_neg_var", False),
    ("A_q10",     True),
    ("A_q25",     True),
    ("A_median",  True),
]
IPA_OP_COLORS = {
    "A_mean":    "#1f77b4",
    "A_min":     "#d62728",
    "A_neg_var": "#2ca02c",
    "A_q10":     "#ff7f0e",
    "A_q25":     "#9467bd",
    "A_median":  "#8c564b",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def agg_mean(v):    return float(np.mean(v))
def agg_min(v):     return float(np.min(v))
def agg_neg_var(v): return float(-np.var(v))
def agg_q10(v):     return float(np.quantile(v, 0.10))
def agg_q25(v):     return float(np.quantile(v, 0.25))
def agg_median(v):  return float(np.median(v))

AGG_FNS = {
    "A_mean": agg_mean, "A_min": agg_min, "A_neg_var": agg_neg_var,
    "A_q10": agg_q10,   "A_q25": agg_q25, "A_median": agg_median,
}


def safe_auc(scores_E, scores_G, higher_memorized: bool) -> float:
    y_true  = [1] * len(scores_E) + [0] * len(scores_G)
    raw     = scores_E + scores_G
    y_score = raw if higher_memorized else [-s for s in raw]
    if len(set(y_true)) < 2 or len(scores_E) == 0 or len(scores_G) == 0:
        return float("nan")
    return roc_auc_score(y_true, y_score)


def load_records(model_tag: str, epoch: int, split: str) -> list:
    path = RESULTS / model_tag / f"run_{epoch}" / split / "records.json"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_mia(model_tag: str, epoch: int, split: str) -> pd.DataFrame:
    path = RESULTS / model_tag / f"run_{epoch}" / split / "mia_summary.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def load_cdd(model_tag: str, epoch: int, split: str) -> pd.DataFrame:
    path = RESULTS / model_tag / f"run_{epoch}" / split / "baseline_results.csv"
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def ipa_scores_per_sample(records: list, agg_fn) -> list:
    by_sample: dict = defaultdict(list)
    for r in records:
        sim = r.get("output_similarity")
        if sim is not None:
            by_sample[r["sample_idx"]].append(sim)
    return [agg_fn(vals) for vals in by_sample.values() if vals]


def available_epochs(model_tag: str) -> list[int]:
    runs = sorted(
        (int(p.name.split("_")[1]) for p in (RESULTS / model_tag).glob("run_*")
         if p.is_dir() and (p / "members" / "records.json").exists())
    )
    return runs


# ── per-model per-epoch analysis ──────────────────────────────────────────────

def analyse_model(model_tag: str) -> dict[str, pd.DataFrame]:
    """Return dict of DataFrames: 'ipa', 'cdd' (optional), 'mia' (optional)."""
    epochs = available_epochs(model_tag)
    if not epochs:
        print(f"  [{model_tag}] no data found, skipping")
        return {}

    print(f"\n{'='*60}")
    print(f"  {MODEL_LABELS.get(model_tag, model_tag)}  —  epochs: {epochs}")
    print(f"{'='*60}")

    ipa_rows, cdd_rows, mia_rows = [], [], []

    for epoch in epochs:
        recs_E = load_records(model_tag, epoch, "members")
        recs_G = load_records(model_tag, epoch, "non_members")
        if not recs_E or not recs_G:
            print(f"  epoch {epoch}: missing records, skip")
            continue

        # ── IPA ───────────────────────────────────────────────────────────
        for op_name, inverted in IPA_OPS:
            fn = AGG_FNS[op_name]
            s_E = ipa_scores_per_sample(recs_E, fn)
            s_G = ipa_scores_per_sample(recs_G, fn)
            mu_E = float(np.mean(s_E)) if s_E else float("nan")
            mu_G = float(np.mean(s_G)) if s_G else float("nan")
            gamma  = mu_G - mu_E      # paper Def. 2: positive = classic PSH
            raw_auc = safe_auc(s_E, s_G, higher_memorized=False)
            corr_auc = (1 - raw_auc) if inverted else raw_auc  # flip if inverted
            ipa_rows.append(dict(
                model=model_tag, epoch=epoch, operator=op_name,
                mu_E=round(mu_E, 6), mu_G=round(mu_G, 6),
                gamma=round(gamma, 6), auc_corrected=round(corr_auc, 4),
            ))

        print(f"  epoch {epoch}  IPA A_mean: "
              f"μ_E={np.mean(ipa_scores_per_sample(recs_E, agg_mean)):.4f}  "
              f"μ_G={np.mean(ipa_scores_per_sample(recs_G, agg_mean)):.4f}")

        # ── CDD ───────────────────────────────────────────────────────────
        df_cdd_E = load_cdd(model_tag, epoch, "members")
        df_cdd_G = load_cdd(model_tag, epoch, "non_members")
        if not df_cdd_E.empty and not df_cdd_G.empty and "cdd_score" in df_cdd_E.columns:
            s_E = df_cdd_E["cdd_score"].dropna().tolist()
            s_G = df_cdd_G["cdd_score"].dropna().tolist()
            mu_E = float(np.mean(s_E))
            mu_G = float(np.mean(s_G))
            auc  = safe_auc(s_E, s_G, higher_memorized=True)
            cdd_rows.append(dict(
                model=model_tag, epoch=epoch,
                mu_E=round(mu_E, 4), mu_G=round(mu_G, 4),
                gamma=round(mu_E - mu_G, 4), auc=round(auc, 4),
                memorized_members=int(df_cdd_E["cdd_memorized"].sum()),
                memorized_non_members=int(df_cdd_G["cdd_memorized"].sum()),
            ))

        # ── MIA ───────────────────────────────────────────────────────────
        mia_E = load_mia(model_tag, epoch, "members")
        mia_G = load_mia(model_tag, epoch, "non_members")
        mia_configs = [
            ("mia_loss", False), ("mia_min_k", True), ("mia_neighborhood", True)
        ]
        for col, higher in mia_configs:
            if col not in mia_E.columns or col not in mia_G.columns:
                continue
            s_E = mia_E[col].dropna().tolist()
            s_G = mia_G[col].dropna().tolist()
            if not s_E or not s_G:
                continue
            auc = safe_auc(s_E, s_G, higher_memorized=higher)
            mia_rows.append(dict(
                model=model_tag, epoch=epoch, method=col, auc=round(auc, 4)
            ))

    dfs: dict[str, pd.DataFrame] = {}
    if ipa_rows:
        dfs["ipa"] = pd.DataFrame(ipa_rows)
        dfs["ipa"].to_csv(OUT_DIR / f"{model_tag}_ipa_per_epoch.csv", index=False)
    if cdd_rows:
        dfs["cdd"] = pd.DataFrame(cdd_rows)
        dfs["cdd"].to_csv(OUT_DIR / f"{model_tag}_cdd_per_epoch.csv", index=False)
    if mia_rows:
        dfs["mia"] = pd.DataFrame(mia_rows)
        dfs["mia"].to_csv(OUT_DIR / f"{model_tag}_mia_per_epoch.csv", index=False)

    return dfs


# ── per-model plots ────────────────────────────────────────────────────────────

def plot_model(model_tag: str, dfs: dict[str, pd.DataFrame]) -> None:
    label = MODEL_LABELS.get(model_tag, model_tag)
    df_ipa = dfs.get("ipa")
    df_cdd = dfs.get("cdd")
    df_mia = dfs.get("mia")

    epochs = sorted(df_ipa["epoch"].unique()) if df_ipa is not None else []
    if not epochs:
        return

    has_cdd = df_cdd is not None and not df_cdd.empty
    has_mia = df_mia is not None and not df_mia.empty
    n_panels = 2 + int(has_cdd) + int(has_mia)

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    col = 0

    # ── IPA corrected AUC ────────────────────────────────────────────────
    ax = axes[col]; col += 1
    for op_name, _ in IPA_OPS:
        sub = df_ipa[df_ipa["operator"] == op_name].sort_values("epoch")
        ax.plot(sub["epoch"], sub["auc_corrected"], marker="o",
                label=op_name, color=IPA_OP_COLORS[op_name], linewidth=1.5)
    ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"IPA AUC — {label}", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Corrected AUC")
    ax.set_xticks(epochs); ax.set_ylim(0.40, 0.75)
    ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)

    # ── IPA gamma ────────────────────────────────────────────────────────
    ax = axes[col]; col += 1
    for op_name, _ in IPA_OPS:
        sub = df_ipa[df_ipa["operator"] == op_name].sort_values("epoch")
        ax.plot(sub["epoch"], sub["gamma"].abs(), marker="o",
                label=op_name, color=IPA_OP_COLORS[op_name], linewidth=1.5)
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_title(f"IPA |γ| — {label}", fontsize=10)
    ax.set_xlabel("Epoch"); ax.set_ylabel("|γ| = |μ(S_G) − μ(S_E)|")
    ax.set_xticks(epochs); ax.legend(fontsize=7, loc="upper left"); ax.grid(True, alpha=0.3)

    # ── CDD ──────────────────────────────────────────────────────────────
    if has_cdd:
        ax = axes[col]; col += 1
        df_cdd_s = df_cdd.sort_values("epoch")
        ax.plot(df_cdd_s["epoch"], df_cdd_s["auc"], "o-",
                color="#e377c2", linewidth=1.8, label="CDD AUC")
        ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
        ax2 = ax.twinx()
        ax2.fill_between(df_cdd_s["epoch"],
                         df_cdd_s["memorized_members"],
                         df_cdd_s["memorized_non_members"],
                         alpha=0.15, color="grey")
        ax2.plot(df_cdd_s["epoch"], df_cdd_s["memorized_members"],
                 "s--", color="#d62728", linewidth=1.2, label="#mem E", alpha=0.7)
        ax2.plot(df_cdd_s["epoch"], df_cdd_s["memorized_non_members"],
                 "^--", color="#1f77b4", linewidth=1.2, label="#mem G", alpha=0.7)
        ax.set_title(f"CDD — {label}", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("AUC", color="#e377c2")
        ax2.set_ylabel("# flagged instances")
        ax.set_xticks(epochs); ax.set_ylim(0.40, 0.70); ax.grid(True, alpha=0.3)
        lines1, l1 = ax.get_legend_handles_labels()
        lines2, l2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, l1 + l2, fontsize=7)

    # ── MIA ──────────────────────────────────────────────────────────────
    if has_mia:
        ax = axes[col]
        mia_colors = {"mia_loss": "#e377c2", "mia_min_k": "#17becf",
                      "mia_neighborhood": "#bcbd22"}
        for method in ["mia_loss", "mia_min_k", "mia_neighborhood"]:
            sub = df_mia[df_mia["method"] == method].sort_values("epoch")
            if sub.empty:
                continue
            ax.plot(sub["epoch"], sub["auc"], "o--",
                    color=mia_colors[method], linewidth=1.5, label=method)
        ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_title(f"MIA AUC — {label}", fontsize=10)
        ax.set_xlabel("Epoch"); ax.set_ylabel("AUC")
        ax.set_xticks(epochs); ax.set_ylim(0.40, 0.85)
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.suptitle(f"{label} — PEARL / CDD / MIA across epochs", fontsize=11, y=1.02)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / f"{model_tag}_full_analysis.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {PLOT_DIR / f'{model_tag}_full_analysis.png'}")


# ── cross-model comparison ────────────────────────────────────────────────────

def plot_model_size_comparison(all_dfs: dict[str, dict]) -> None:
    """
    Two figures:
    1) Side-by-side bar chart at epoch 1 and epoch 10:
       IPA A_mean AUC + CDD AUC + MIA min_k AUC per model
    2) Line plot: IPA A_mean AUC across ALL available epochs, one line per model
    """
    models_present = [m for m in MODEL_ORDER if m in all_dfs and all_dfs[m]]

    # ── collect comparison rows ───────────────────────────────────────────
    comp_rows = []
    for model in models_present:
        dfs = all_dfs[model]
        df_ipa = dfs.get("ipa")
        df_cdd = dfs.get("cdd")
        df_mia = dfs.get("mia")
        epochs = sorted(df_ipa["epoch"].unique()) if df_ipa is not None else []
        for ep in epochs:
            row = dict(model=model, epoch=ep,
                       params_m=MODEL_PARAMS.get(model, 0),
                       label=MODEL_LABELS.get(model, model))
            if df_ipa is not None:
                sub = df_ipa[(df_ipa["model"] == model) &
                             (df_ipa["epoch"] == ep) &
                             (df_ipa["operator"] == "A_mean")]
                row["ipa_auc"] = sub["auc_corrected"].values[0] if len(sub) else np.nan
                row["ipa_gamma"] = abs(sub["gamma"].values[0]) if len(sub) else np.nan
            if df_cdd is not None:
                sub = df_cdd[(df_cdd["model"] == model) & (df_cdd["epoch"] == ep)]
                row["cdd_auc"] = sub["auc"].values[0] if len(sub) else np.nan
            if df_mia is not None:
                sub = df_mia[(df_mia["model"] == model) & (df_mia["epoch"] == ep) &
                             (df_mia["method"] == "mia_min_k")]
                row["mia_auc"] = sub["auc"].values[0] if len(sub) else np.nan
            comp_rows.append(row)

    df_comp = pd.DataFrame(comp_rows)
    df_comp.to_csv(OUT_DIR / "model_size_comparison.csv", index=False)

    # ── Figure 1: bar chart at epoch 1 and 10 ────────────────────────────
    metrics = [("ipa_auc", "IPA A_mean (corrected AUC)"),
               ("cdd_auc", "CDD AUC"),
               ("mia_auc", "MIA min_k AUC")]
    compare_epochs = [ep for ep in [1, 10] if ep in df_comp["epoch"].values]

    fig, axes = plt.subplots(1, len(compare_epochs),
                             figsize=(6 * len(compare_epochs), 5), sharey=True)
    if len(compare_epochs) == 1:
        axes = [axes]

    bar_colors = {"ipa_auc": "#1f77b4", "cdd_auc": "#e377c2", "mia_auc": "#17becf"}
    x = np.arange(len(models_present))
    bar_w = 0.25

    for ax_i, ep in enumerate(compare_epochs):
        ax = axes[ax_i]
        sub = df_comp[df_comp["epoch"] == ep]
        for m_i, (metric, mlabel) in enumerate(metrics):
            vals = [sub[sub["model"] == m][metric].values[0]
                    if len(sub[sub["model"] == m]) and metric in sub.columns else np.nan
                    for m in models_present]
            offset = (m_i - 1) * bar_w
            bars = ax.bar(x + offset, vals, bar_w,
                          label=mlabel, color=bar_colors[metric], alpha=0.82,
                          edgecolor="black", linewidth=0.5)
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.text(bar.get_x() + bar.get_width()/2, v + 0.003,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=6.5)
        ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5, label="Random")
        ax.set_title(f"Epoch {ep}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS.get(m, m) for m in models_present],
                           fontsize=9)
        ax.set_ylim(0.40, 0.80)
        ax.set_ylabel("AUC", fontsize=10)
        ax.grid(True, alpha=0.25, axis="y")
        if ax_i == 0:
            ax.legend(fontsize=8, loc="upper left")

    plt.suptitle("Detection AUC by model size — Pythia family", fontsize=12)
    plt.tight_layout()
    fig.savefig(PLOT_DIR / "model_size_comparison_bars.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ── Figure 2: line plot across all epochs (IPA A_mean) ───────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    for model in models_present:
        sub = df_comp[df_comp["model"] == model].sort_values("epoch")
        col = MODEL_COLORS.get(model, "grey")
        lbl = MODEL_LABELS.get(model, model)
        axes2[0].plot(sub["epoch"], sub["ipa_auc"], "o-", color=col,
                      linewidth=2, label=lbl, markersize=5)
        axes2[1].plot(sub["epoch"], sub["ipa_gamma"], "o-", color=col,
                      linewidth=2, label=lbl, markersize=5)

    axes2[0].axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5)
    axes2[0].set_xlabel("Epoch"); axes2[0].set_ylabel("IPA A_mean corrected AUC")
    axes2[0].set_title("IPA A_mean AUC across epochs — model size effect")
    axes2[0].legend(fontsize=9); axes2[0].grid(True, alpha=0.3)
    axes2[0].set_ylim(0.40, 0.80)

    axes2[1].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    axes2[1].set_xlabel("Epoch"); axes2[1].set_ylabel("|γ| = |μ(S_G) − μ(S_E)|")
    axes2[1].set_title("IPA A_mean gap |γ| across epochs — model size effect")
    axes2[1].legend(fontsize=9); axes2[1].grid(True, alpha=0.3)

    plt.suptitle("Scaling behaviour — Pythia family (PEARL IPA A_mean)", fontsize=12)
    plt.tight_layout()
    fig2.savefig(PLOT_DIR / "model_size_comparison_lines.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig2)

    # ── Figure 3: CDD across epochs (models with full epoch sweep) ────────
    models_with_cdd = [m for m in models_present
                       if all_dfs[m].get("cdd") is not None
                       and not all_dfs[m]["cdd"].empty]
    if models_with_cdd:
        fig3, ax3 = plt.subplots(figsize=(9, 5))
        for model in models_with_cdd:
            sub = all_dfs[model]["cdd"].sort_values("epoch")
            col = MODEL_COLORS.get(model, "grey")
            ax3.plot(sub["epoch"], sub["auc"], "o-", color=col,
                     linewidth=2, label=MODEL_LABELS.get(model, model))
        ax3.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.5,
                    label="Random (0.5)")
        ax3.set_xlabel("Epoch"); ax3.set_ylabel("CDD AUC")
        ax3.set_title("CDD peakedness AUC across epochs — model size comparison")
        ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3); ax3.set_ylim(0.40, 0.75)
        plt.tight_layout()
        fig3.savefig(PLOT_DIR / "model_size_cdd_lines.png",
                     dpi=150, bbox_inches="tight")
        plt.close(fig3)

    # ── Figure 4: model size (x=params) at epoch 1 and 10 ────────────────
    fig4, axes4 = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    metrics_line = [("ipa_auc", "IPA A_mean", "#1f77b4"),
                    ("cdd_auc", "CDD",         "#e377c2"),
                    ("mia_auc", "MIA min_k",   "#17becf")]
    xvals = sorted(set(df_comp["params_m"].dropna()))

    for ax_i, ep in enumerate([1, 10]):
        ax = axes4[ax_i]
        sub_ep = df_comp[df_comp["epoch"] == ep].sort_values("params_m")
        for metric, mlabel, mcol in metrics_line:
            y = sub_ep[metric].values if metric in sub_ep.columns else []
            x = sub_ep["params_m"].values
            valid = ~np.isnan(y.astype(float))
            if valid.any():
                ax.plot(x[valid], y[valid], "o-", color=mcol,
                        linewidth=2, markersize=7, label=mlabel)
                for xi, yi in zip(x[valid], y[valid]):
                    ax.annotate(f"{yi:.3f}", (xi, yi),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=7.5)
        ax.axhline(0.5, color="black", lw=0.8, ls="--", alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("Model size (M parameters, log scale)", fontsize=10)
        ax.set_ylabel("AUC", fontsize=10)
        ax.set_title(f"Epoch {ep} — AUC vs model size", fontsize=11)
        ax.set_xticks([70, 410, 1400, 2800])
        ax.set_xticklabels(["70M", "410M", "1.4B", "2.8B"])
        ax.set_ylim(0.40, 0.80)
        ax.grid(True, alpha=0.3, which="both")
        if ax_i == 0:
            ax.legend(fontsize=9)

    plt.suptitle("AUC vs model size at epoch 1 and epoch 10 — Pythia family",
                 fontsize=12)
    plt.tight_layout()
    fig4.savefig(PLOT_DIR / "model_size_auc_vs_params.png",
                 dpi=150, bbox_inches="tight")
    plt.close(fig4)

    print(f"\nComparison plots saved to {PLOT_DIR}")
    return df_comp


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None,
                   help="Model tags to analyse (default: auto-detect from results/)")
    return p.parse_args()


def main():
    args = parse_args()
    if args.models:
        models = args.models
    else:
        models = [d.name for d in sorted(RESULTS.iterdir())
                  if d.is_dir() and d.name != "logs"
                  and any(d.glob("run_*/members/records.json"))]
    models = [m for m in MODEL_ORDER if m in models] + \
             [m for m in models if m not in MODEL_ORDER]

    print(f"Models to analyse: {models}")
    all_dfs: dict[str, dict] = {}
    for model in models:
        dfs = analyse_model(model)
        if dfs:
            all_dfs[model] = dfs
            plot_model(model, dfs)

    if len(all_dfs) > 1:
        print("\n── Cross-model comparison ──")
        df_comp = plot_model_size_comparison(all_dfs)
        print("\n=== Model size comparison (epoch 1 and 10, IPA A_mean) ===")
        piv = df_comp[df_comp["epoch"].isin([1, 10])][
            ["label", "epoch", "ipa_auc", "cdd_auc", "mia_auc", "ipa_gamma"]
        ].sort_values(["epoch", "label"])
        print(piv.to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()
