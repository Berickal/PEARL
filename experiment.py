"""
Experiment: Input Cosine Similarity vs Output Cosine Similarity + MIA scores.

For each sample in the benchmark:
  1. Compute MIA scores on the original input (Loss, Min-K%, Neighborhood).
  2. For each similarity range bucket, generate N modified inputs whose cosine
     similarity to the original falls within the bucket.
  3. Generate model outputs for the original and every modified input.
  4. Compute output cosine similarity between original and modified outputs.
  5. Record everything in a flat list of dicts.

The [0.90-1.00] range candidates are reused as the neighborhood for the
Neighborhood Attack MIA — no extra generation needed.

Outputs (written to --output-dir):
  records.json          — full per-modification records
  summary.csv           — mean/std output_sim per input-sim bucket
  mia_summary.csv       — per-sample MIA scores
  similarity_curve.png  — line chart

Member / Non-member comparison
-------------------------------
    # Run on both splits, model loaded only once
    python experiment.py --config config.yml --dataset-split both

    # Or one at a time
    python experiment.py --config config.yml --dataset-split leak
    python experiment.py --config config.yml --dataset-split unleak

    Results land in:
        results/experiment/members/
        results/experiment/non_members/
        results/experiment/comparison.png

Two-phase workflow (recommended for large datasets)
----------------------------------------------------
    # Phase 1 — generate + save perturbations (no GPU needed)
    python -m perturbation.preprocess --config config.yml \\
        --dataset-split leak   --output data/perturbations_members.json
    python -m perturbation.preprocess --config config.yml \\
        --dataset-split unleak --output data/perturbations_non_members.json

    # Phase 2 — inference + MIA using pre-generated perturbations
    python experiment.py --config config.yml --dataset-split both \\
        --members-perturbations-file     data/perturbations_members.json \\
        --non-members-perturbations-file data/perturbations_non_members.json

Usage
-----
    python experiment.py [--config config.yml] [--output-dir results/]
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── path setup ───────────────────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils import load_config
from logging_config import setup_logging
from data_loader import load_dataset, load_split_benchmark, Benchmark
from evaluation.similarity import calculate_cosine_similarity
from evaluation.reporting import (
    range_label,
    aggregate_by_range,
    save_records_json,
    save_summary_csv,
    save_mia_summary_csv,
    plot_similarity_curve,
    plot_comparison_curve,
)
from perturbation.range_perturbation import generate_for_range

logger = logging.getLogger(__name__)

# ── defaults ──────────────────────────────────────────────────────────────────
DEFAULT_RANGES: List[Tuple[float, float]] = [
    (0.90, 1.00),
    (0.80, 0.90),
    (0.70, 0.80),
    (0.60, 0.70),
    (0.50, 0.60),
]
DEFAULT_N_PER_RANGE     = 5
DEFAULT_BUDGET          = 300
DEFAULT_SIMILARITY_METRIC = 'cosine'
DEFAULT_OUTPUT_DIR      = 'results/experiment'
DEFAULT_MIA_MIN_K       = 0.2
_NEIGHBORHOOD_RANGE     = (0.90, 1.00)


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_ranges(raw: List[List[float]]) -> List[Tuple[float, float]]:
    """[[1.0, 0.9], ...] → [(0.9, 1.0), ...] sorted low→high."""
    result = []
    for pair in raw:
        lo, hi = sorted(pair)
        result.append((round(lo, 2), round(hi, 2)))
    return result


def _load_perturbations_file(path: str) -> Dict[int, Dict]:
    """Load a pre-generated perturbations JSON, keyed by sample_idx."""
    logger.info(f"Loading pre-generated perturbations from: {path}")
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Perturbations file is corrupted or truncated: {path}\n"
            f"  JSON error at line {exc.lineno}, col {exc.colno}: {exc.msg}\n"
            f"  This usually means the preprocess run was killed mid-write.\n"
            f"  Re-run with --resume to recover already-processed samples:\n"
            f"    python -m perturbation.preprocess --config config.yml "
            f"--output {path} --resume"
        ) from exc
    meta    = data.get('metadata', {})
    samples = data.get('samples',  [])
    lookup  = {s['sample_idx']: s for s in samples}
    logger.info(
        f"  → {len(lookup)} samples | "
        f"dataset={meta.get('dataset', '?')} | "
        f"generated={meta.get('generated_at', '?')}"
    )
    return lookup


def _load_model(config: Dict[str, Any]):
    """Load ModelGenerator + optional MIAScorer from config."""
    from inference.generator import ModelGenerator
    from inference.mia import MIAScorer

    inf_cfg   = config.get('inference', {})
    model_cfg = config.get('model',     {})
    mia_cfg   = config.get('mia',       {})

    generator = ModelGenerator(
        model_name_or_path=model_cfg.get('name_or_path', 'EleutherAI/pythia-70m'),
        device=model_cfg.get('device', 'auto'),
        torch_dtype=model_cfg.get('torch_dtype', 'bfloat16'),
        max_new_tokens=inf_cfg.get('max_new_tokens', 100),
        do_sample=inf_cfg.get('do_sample', False),
        batch_size=inf_cfg.get('batch_size', 8),
    )

    mia_scorer = None
    if mia_cfg.get('enabled', True):
        mia_scorer = MIAScorer(
            model=generator.model,
            tokenizer=generator.tokenizer,
            device=generator.device,
            max_length=model_cfg.get('max_length', inf_cfg.get('max_length', 512)),
            min_k=mia_cfg.get('min_k', DEFAULT_MIA_MIN_K),
        )
        logger.info("MIA scorer initialised (Loss, Min-K%, Neighborhood Attack).")

    return generator, mia_scorer


# ── core per-sample loop ──────────────────────────────────────────────────────

def _run_on_benchmark(
    benchmark: Benchmark,
    generator,
    mia_scorer,
    similarity_ranges: List[Tuple[float, float]],
    n_per_range: int,
    budget: int,
    sim_metric: str,
    seed: int,
    precomputed: Optional[Dict[int, Dict]] = None,
) -> List[Dict[str, Any]]:
    """
    Run the experiment loop over every sample in *benchmark*.

    Model objects are passed in so they can be shared across multiple benchmark
    runs (member + non-member) without reloading.
    """
    all_records: List[Dict[str, Any]] = []
    total = len(benchmark.samples)

    for sample_idx, sample in enumerate(benchmark.samples):
        t0 = time.time()
        logger.info(
            f"[{sample_idx + 1}/{total}] "
            f"input: {sample.input[:70].replace(chr(10), ' ')}..."
        )

        # ── MIA: Loss + Min-K (no generation needed) ─────────────────────────
        mia_loss = mia_min_k_score = None
        if mia_scorer is not None:
            mia_loss        = mia_scorer.compute_loss(sample.input)
            mia_min_k_score = mia_scorer.compute_min_k(sample.input)
            logger.debug(f"  mia_loss={mia_loss}  mia_min_k={mia_min_k_score}")

        # ── generate original output ──────────────────────────────────────────
        original_output = generator.generate([sample.input])[0]

        # ── per-range loop ────────────────────────────────────────────────────
        neighborhood_texts: List[str] = []
        sample_records: List[Dict[str, Any]] = []

        if precomputed is not None and sample_idx in precomputed:
            pre = precomputed[sample_idx]
            range_iter = []
            for label, cands in pre['ranges'].items():
                parts = label.strip('[]').split('-')
                lo, hi = float(parts[0]), float(parts[1])
                range_iter.append((lo, hi, label, cands))
            use_precomputed = True
        else:
            range_iter = [
                (lo, hi, range_label(lo, hi), None)
                for lo, hi in similarity_ranges
            ]
            use_precomputed = False

        for sim_low, sim_high, label, pre_candidates in range_iter:
            if use_precomputed:
                candidates = pre_candidates or []
                if candidates:
                    logger.info(f"  range {label}: {len(candidates)} pre-generated candidates.")
                else:
                    logger.warning(f"  range {label}: 0 candidates in file — skipping.")
            else:
                logger.info(f"  range {label}: generating {n_per_range} modifications...")
                candidates = generate_for_range(
                    original_text=sample.input,
                    sim_low=sim_low,
                    sim_high=sim_high,
                    n=n_per_range,
                    similarity_metric=sim_metric,
                    seed=seed + sample_idx * 1000,
                    budget=budget,
                )

            if not candidates:
                logger.warning(f"  range {label}: 0 candidates — skipping.")
                continue

            if (sim_low, sim_high) == _NEIGHBORHOOD_RANGE:
                neighborhood_texts = [c['modified_text'] for c in candidates]

            modified_texts   = [c['modified_text'] for c in candidates]
            modified_outputs = generator.generate(modified_texts)

            for candidate, modified_output in zip(candidates, modified_outputs):
                output_sim = calculate_cosine_similarity(original_output, modified_output)
                sample_records.append({
                    'sample_idx':        sample_idx,
                    'range_label':       label,
                    'sim_low':           sim_low,
                    'sim_high':          sim_high,
                    'input_similarity':  candidate['similarity'],
                    'output_similarity': round(output_sim, 6),
                    'perturbation_type': candidate['perturbation_type'],
                    'mia_loss':          mia_loss,
                    'mia_min_k':         mia_min_k_score,
                    'mia_neighborhood':  None,   # backfilled below
                    'original_input':    sample.input,
                    'modified_input':    candidate['modified_text'],
                    'original_output':   original_output,
                    'modified_output':   modified_output,
                })

        # ── MIA: Neighborhood Attack ──────────────────────────────────────────
        mia_neighborhood: Optional[float] = None
        if mia_scorer is not None:
            mia_neighborhood = mia_scorer.compute_neighborhood_score(
                sample.input, neighborhood_texts
            )
            logger.debug(f"  mia_neighborhood={mia_neighborhood}")

        for r in sample_records:
            r['mia_neighborhood'] = mia_neighborhood

        all_records.extend(sample_records)

        elapsed = time.time() - t0
        logger.info(
            f"  done in {elapsed:.1f}s | "
            f"records this sample: {len(sample_records)} | "
            f"total so far: {len(all_records)}"
        )

    return all_records


# ── public experiment API ─────────────────────────────────────────────────────

def run_experiment(
    config: Dict[str, Any],
    perturbations_file: Optional[str] = None,
    benchmark: Optional[Benchmark] = None,
) -> List[Dict[str, Any]]:
    """
    Run the experiment on a single benchmark (backward-compatible entry point).

    If *benchmark* is not given, it is loaded from ``config['dataset']``.
    """
    exp_cfg = config.get('experiment', {})
    mia_cfg = config.get('mia',        {})

    similarity_ranges = _parse_ranges(
        exp_cfg.get('similarity_ranges', [[hi, lo] for lo, hi in DEFAULT_RANGES])
    )
    n_per_range = exp_cfg.get('n_per_range',          DEFAULT_N_PER_RANGE)
    budget      = exp_cfg.get('perturbation_budget',  DEFAULT_BUDGET)
    sim_metric  = exp_cfg.get('similarity_metric',    DEFAULT_SIMILARITY_METRIC)
    seed        = exp_cfg.get('seed',                 42)

    if mia_cfg.get('enabled', True) and _NEIGHBORHOOD_RANGE not in similarity_ranges:
        similarity_ranges = [_NEIGHBORHOOD_RANGE] + similarity_ranges

    precomputed = _load_perturbations_file(perturbations_file) if perturbations_file else None

    if benchmark is None:
        logger.info("Loading dataset...")
        benchmark = load_dataset(config)
        logger.info(f"  → {len(benchmark.samples)} samples from '{benchmark.name}'")

    generator, mia_scorer = _load_model(config)

    return _run_on_benchmark(
        benchmark, generator, mia_scorer,
        similarity_ranges, n_per_range, budget, sim_metric, seed,
        precomputed=precomputed,
    )


def run_experiment_splits(
    config: Dict[str, Any],
    splits: List[str],
    perturbations_files: Optional[Dict[str, str]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run the experiment on one or more named dataset splits.

    The model is loaded **once** and shared across all splits, which is
    important when running members and non-members back-to-back.

    Parameters
    ----------
    config :
        Loaded config dict.
    splits :
        List of split names to run, e.g. ``['leak', 'unleak']``.
        Each name must be a key under ``config['datasets']``.
    perturbations_files :
        Optional ``{split_name: path}`` mapping of pre-generated perturbation
        files (produced by ``perturbation/preprocess.py``).

    Returns
    -------
    Dict[str, List[Dict]]
        Keyed by split name, values are the flat record lists.
    """
    exp_cfg = config.get('experiment', {})
    mia_cfg = config.get('mia',        {})

    similarity_ranges = _parse_ranges(
        exp_cfg.get('similarity_ranges', [[hi, lo] for lo, hi in DEFAULT_RANGES])
    )
    n_per_range = exp_cfg.get('n_per_range',          DEFAULT_N_PER_RANGE)
    budget      = exp_cfg.get('perturbation_budget',  DEFAULT_BUDGET)
    sim_metric  = exp_cfg.get('similarity_metric',    DEFAULT_SIMILARITY_METRIC)
    seed        = exp_cfg.get('seed',                 42)

    if mia_cfg.get('enabled', True) and _NEIGHBORHOOD_RANGE not in similarity_ranges:
        similarity_ranges = [_NEIGHBORHOOD_RANGE] + similarity_ranges

    logger.info("Loading model (shared across splits)...")
    generator, mia_scorer = _load_model(config)

    results: Dict[str, List[Dict[str, Any]]] = {}

    for split in splits:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Split: {split}")
        logger.info(f"{'=' * 60}")

        benchmark = load_split_benchmark(config, split)
        logger.info(f"  → {len(benchmark.samples)} samples from '{benchmark.name}'")

        precomputed = None
        if perturbations_files and split in perturbations_files:
            precomputed = _load_perturbations_file(perturbations_files[split])

        results[split] = _run_on_benchmark(
            benchmark, generator, mia_scorer,
            similarity_ranges, n_per_range, budget, sim_metric, seed,
            precomputed=precomputed,
        )

    return results


# ── output ────────────────────────────────────────────────────────────────────

def save_results(
    records: List[Dict[str, Any]],
    output_dir: str,
    model_name: str = '',
    label: str = '',
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    save_records_json(records, str(out / 'records.json'))

    aggregated = aggregate_by_range(records)
    save_summary_csv(aggregated, str(out / 'summary.csv'))
    save_mia_summary_csv(records, str(out / 'mia_summary.csv'))

    title_parts = ['Output Similarity vs Input Similarity Range']
    if label:
        title_parts.append(label)
    if model_name:
        title_parts.append(model_name)
    plot_similarity_curve(
        aggregated, str(out / 'similarity_curve.png'),
        title='\n'.join(title_parts),
    )

    print("\n" + "=" * 72)
    if label:
        print(f"  {label}")
    print(f"{'Range':<18} {'N':>5} {'Mean Output Sim':>16} {'Std':>8}")
    print("-" * 72)
    for lbl, stats in aggregated.items():
        print(
            f"{lbl:<18} {stats['n']:>5} "
            f"{stats['mean_output_sim']:>16.4f} "
            f"{stats['std_output_sim']:>8.4f}"
        )
    print("=" * 72)

    mia_rows = {}
    for r in records:
        idx = r['sample_idx']
        if idx not in mia_rows and r.get('mia_loss') is not None:
            mia_rows[idx] = r
    if mia_rows:
        print()
        print(f"{'Sample':>8} {'Loss':>10} {'Min-K%':>10} {'Neighborhood':>14}")
        print("-" * 46)
        for idx in sorted(mia_rows):
            r   = mia_rows[idx]
            nbr = r['mia_neighborhood']
            print(
                f"{idx:>8} "
                f"{r['mia_loss']:>10.4f} "
                f"{r['mia_min_k']:>10.4f} "
                f"{nbr if nbr is not None else 'N/A':>14}"
            )

    print(f"\nResults saved to: {output_dir}")


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Run input/output cosine-similarity range experiment with MIA'
    )
    parser.add_argument('--config',     type=str, default='config.yml')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument(
        '--model', type=str, default=None,
        metavar='PATH_OR_ID',
        help=(
            'HuggingFace model ID or local path to load for this run. '
            'Overrides model.name_or_path in config.yml. '
            'Use this to iterate over fine-tuned checkpoints, e.g. '
            'checkpoints/EleutherAI_pythia-70m_fine_tuned/checkpoint-epoch-3'
        ),
    )
    parser.add_argument(
        '--dataset-split',
        type=str, default=None,
        choices=['leak', 'unleak', 'both'],
        help=(
            'Dataset split to use.  '
            '"leak" = members only (datasets.leak in config);  '
            '"unleak" = non-members only (datasets.unleak);  '
            '"both" = run both, load model once, produce comparison plot.  '
            'Omit to use the top-level dataset key (backward compat).'
        ),
    )
    parser.add_argument(
        '--perturbations-file', type=str, default=None,
        help='Pre-generated perturbations JSON (single-split runs only).',
    )
    parser.add_argument(
        '--members-perturbations-file', type=str, default=None,
        help='Pre-generated perturbations for the "leak" split (--dataset-split both).',
    )
    parser.add_argument(
        '--non-members-perturbations-file', type=str, default=None,
        help='Pre-generated perturbations for the "unleak" split (--dataset-split both).',
    )
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    )
    parser.add_argument('--log-file', type=str, default=None)
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(_SRC_DIR / config_path)

    logger.info(f"Loading config: {config_path}")
    config = load_config(config_path)

    # --model overrides model.name_or_path from config
    if args.model:
        config.setdefault('model', {})['name_or_path'] = args.model
        logger.info(f"Model overridden via --model: {args.model}")

    model_name = config.get('model', {}).get('name_or_path', '')
    base_dir   = args.output_dir or config.get('experiment', {}).get(
        'output_dir', DEFAULT_OUTPUT_DIR
    )

    if args.dataset_split == 'both':
        # ── Both splits — model loaded once ──────────────────────────────────
        pert_files: Dict[str, str] = {}
        if args.members_perturbations_file:
            pert_files['leak']   = args.members_perturbations_file
        if args.non_members_perturbations_file:
            pert_files['unleak'] = args.non_members_perturbations_file

        results = run_experiment_splits(
            config,
            splits=['leak', 'unleak'],
            perturbations_files=pert_files or None,
        )

        members_dir     = str(Path(base_dir) / 'members')
        non_members_dir = str(Path(base_dir) / 'non_members')

        save_results(results['leak'],   members_dir,     model_name, label='Members')
        save_results(results['unleak'], non_members_dir, model_name, label='Non-members')

        members_agg     = aggregate_by_range(results['leak'])
        non_members_agg = aggregate_by_range(results['unleak'])
        comparison_path = str(Path(base_dir) / 'comparison.png')
        plot_comparison_curve(
            members_agg, non_members_agg,
            comparison_path,
            title=f'Members vs Non-members — Output Similarity\n{model_name}',
        )
        print(f"\nComparison plot → {comparison_path}")

    elif args.dataset_split in ('leak', 'unleak'):
        # ── Single named split ────────────────────────────────────────────────
        sub = 'members' if args.dataset_split == 'leak' else 'non_members'
        output_dir = str(Path(base_dir) / sub)
        label      = 'Members' if args.dataset_split == 'leak' else 'Non-members'

        benchmark = load_split_benchmark(config, args.dataset_split)
        records   = run_experiment(
            config,
            perturbations_file=args.perturbations_file,
            benchmark=benchmark,
        )
        save_results(records, output_dir, model_name, label=label)

    else:
        # ── Backward compat: use config['dataset'] ────────────────────────────
        records = run_experiment(config, perturbations_file=args.perturbations_file)
        save_results(records, base_dir, model_name)


if __name__ == '__main__':
    main()
