"""
Reporting utilities: aggregate experiment records, save CSV/JSON, produce plots.
"""
import csv
import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# Canonical display order (highest → lowest input similarity)
_RANGE_ORDER = [
    "[0.90-1.00]",
    "[0.80-0.90]",
    "[0.70-0.80]",
    "[0.60-0.70]",
    "[0.50-0.60]",
]


def range_label(sim_low: float, sim_high: float) -> str:
    """Canonical string label for a similarity range bucket."""
    return f"[{sim_low:.2f}-{sim_high:.2f}]"


def aggregate_by_range(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate output-similarity stats per input-similarity bucket.

    Returns a dict keyed by range_label, with keys:
        n, mean, std, median, min, max, input_sim_mean
    """
    output_sims: Dict[str, List[float]] = defaultdict(list)
    input_sims: Dict[str, List[float]] = defaultdict(list)

    for r in records:
        label = r['range_label']
        output_sims[label].append(float(r['output_similarity']))
        input_sims[label].append(float(r['input_similarity']))

    result = {}
    for label in _RANGE_ORDER:
        sims = output_sims.get(label, [])
        in_sims = input_sims.get(label, [])
        if not sims:
            continue
        result[label] = {
            'n': len(sims),
            'mean_output_sim': round(statistics.mean(sims), 6),
            'std_output_sim': round(statistics.stdev(sims) if len(sims) > 1 else 0.0, 6),
            'median_output_sim': round(statistics.median(sims), 6),
            'min_output_sim': round(min(sims), 6),
            'max_output_sim': round(max(sims), 6),
            'mean_input_sim': round(statistics.mean(in_sims), 6) if in_sims else None,
        }

    return result


def _strip_surrogates(obj: Any) -> Any:
    """Recursively remove lone surrogate code points from all strings."""
    if isinstance(obj, str):
        return obj.encode('utf-8', errors='ignore').decode('utf-8')
    if isinstance(obj, dict):
        return {k: _strip_surrogates(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_surrogates(v) for v in obj]
    return obj


def save_records_json(records: List[Dict[str, Any]], path: str) -> None:
    """Save full per-sample records as a JSON file."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(_strip_surrogates(records), f, indent=2, ensure_ascii=False)
    logger.info(f"Saved {len(records)} records → {path}")


def save_summary_csv(aggregated: Dict[str, Dict[str, Any]], path: str) -> None:
    """Save one-row-per-range summary statistics as CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        'range', 'n',
        'mean_output_sim', 'std_output_sim', 'median_output_sim',
        'min_output_sim', 'max_output_sim', 'mean_input_sim',
    ]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, stats in aggregated.items():
            writer.writerow({'range': label, **stats})
    logger.info(f"Saved summary ({len(aggregated)} ranges) → {path}")


def save_mia_summary_csv(records: List[Dict[str, Any]], path: str) -> None:
    """
    Save one row per sample with its three MIA scores.

    Reads mia_loss, mia_min_k, mia_neighborhood from the first record
    seen for each sample_idx (they are the same across all modifications
    of the same sample).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    seen: Dict[int, Dict[str, Any]] = {}
    for r in records:
        idx = r.get('sample_idx')
        if idx not in seen and any(
            k in r for k in ('mia_loss', 'mia_min_k', 'mia_neighborhood')
        ):
            seen[idx] = {
                'sample_idx': idx,
                'mia_loss': r.get('mia_loss'),
                'mia_min_k': r.get('mia_min_k'),
                'mia_neighborhood': r.get('mia_neighborhood'),
            }

    if not seen:
        logger.info("No MIA scores found in records; skipping MIA summary CSV.")
        return

    fieldnames = ['sample_idx', 'mia_loss', 'mia_min_k', 'mia_neighborhood']
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(seen.values(), key=lambda x: x['sample_idx']):
            writer.writerow(row)
    logger.info(f"Saved MIA summary ({len(seen)} samples) → {path}")


def plot_similarity_curve(
    aggregated: Dict[str, Dict[str, Any]],
    path: str,
    title: Optional[str] = None,
) -> None:
    """
    Line chart: input similarity range (x) vs mean output similarity ± std (y).
    Requires matplotlib; logs a warning and returns silently if unavailable.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping plot")
        return

    labels = [l for l in _RANGE_ORDER if l in aggregated]
    if not labels:
        logger.warning("No data to plot")
        return

    means = [aggregated[l]['mean_output_sim'] for l in labels]
    stds = [aggregated[l]['std_output_sim'] for l in labels]
    ns = [aggregated[l]['n'] for l in labels]

    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(x, means, yerr=stds, marker='o', linewidth=2, capsize=6,
                color='steelblue', ecolor='lightsteelblue', label='mean ± std')
    ax.fill_between(
        x,
        [m - s for m, s in zip(means, stds)],
        [m + s for m, s in zip(means, stds)],
        alpha=0.15, color='steelblue',
    )

    # Annotate n
    for xi, n in zip(x, ns):
        ax.annotate(f'n={n}', xy=(xi, 0.02), ha='center', fontsize=8, color='grey',
                    xycoords=('data', 'axes fraction'))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Input Cosine Similarity Range', fontsize=12)
    ax.set_ylabel('Mean Output Cosine Similarity', fontsize=12)
    ax.set_title(title or 'Output Similarity vs Input Similarity Range', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend()

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Plot saved → {path}")


def plot_comparison_curve(
    members_agg: Dict[str, Dict[str, Any]],
    non_members_agg: Dict[str, Dict[str, Any]],
    path: str,
    title: Optional[str] = None,
) -> None:
    """
    Line chart comparing output-similarity curves for members vs non-members.

    Members (training data) are expected to maintain high output similarity
    even as input similarity decreases — the gap between the two curves is
    the memorisation signal.
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not available — skipping comparison plot")
        return

    all_labels = [l for l in _RANGE_ORDER if l in members_agg or l in non_members_agg]
    if not all_labels:
        logger.warning("No data to plot for comparison")
        return

    def _series(agg):
        means, stds = [], []
        for l in all_labels:
            if l in agg:
                means.append(agg[l]['mean_output_sim'])
                stds.append(agg[l]['std_output_sim'])
            else:
                means.append(float('nan'))
                stds.append(float('nan'))
        return np.array(means), np.array(stds)

    m_means, m_stds = _series(members_agg)
    nm_means, nm_stds = _series(non_members_agg)
    x = np.arange(len(all_labels))

    fig, ax = plt.subplots(figsize=(11, 6))

    # Members
    ax.errorbar(x, m_means, yerr=m_stds, marker='o', linewidth=2, capsize=5,
                color='steelblue', ecolor='lightsteelblue', label='Members (leak)')
    ax.fill_between(x, m_means - m_stds, m_means + m_stds, alpha=0.12, color='steelblue')

    # Non-members
    ax.errorbar(x, nm_means, yerr=nm_stds, marker='s', linewidth=2, capsize=5,
                color='tomato', ecolor='lightsalmon', label='Non-members (unleak)',
                linestyle='--')
    ax.fill_between(x, nm_means - nm_stds, nm_means + nm_stds, alpha=0.12, color='tomato')

    ax.set_xticks(x)
    ax.set_xticklabels(all_labels, rotation=30, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_xlabel('Input Cosine Similarity Range', fontsize=12)
    ax.set_ylabel('Mean Output Cosine Similarity', fontsize=12)
    ax.set_title(title or 'Members vs Non-members: Output Similarity Curve', fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Comparison plot saved → {path}")
