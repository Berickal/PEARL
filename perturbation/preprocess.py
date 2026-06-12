"""
Perturbation pre-processing — run BEFORE the main experiment.

Generates all modified inputs for every sample and saves them to a JSON file
WITHOUT loading the language model.  The saved file can be passed to
experiment.py via --perturbations-file so perturbation generation is skipped
on every subsequent run.

Usage
-----
    # Full dataset, sequential (from src_v3/)
    python -m perturbation.preprocess --config config.yml

    # Parallel across 8 CPU cores
    python -m perturbation.preprocess --config config.yml --workers 8

    # Member / non-member splits
    python -m perturbation.preprocess --config config.yml \\
        --dataset-split leak   --output data/perturbations_members.json
    python -m perturbation.preprocess --config config.yml \\
        --dataset-split unleak --output data/perturbations_non_members.json

    # Resume an interrupted run (skips already-processed samples)
    python -m perturbation.preprocess --config config.yml --resume

    # Test on 50 samples
    python -m perturbation.preprocess --config config.yml --limit 50

    # Then run the experiment using the saved perturbations
    python experiment.py \\
        --config config.yml \\
        --perturbations-file data/perturbations.json

Parallel notes
--------------
- Uses ProcessPoolExecutor (process-based) — safe for CPU-bound work and nlpaug.
- Results are written once at the end; use --resume for crash recovery:
    python -m perturbation.preprocess --config config.yml --workers 8 --resume
  Any samples already in the output file are skipped on the next run.
"""
import argparse
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils import load_config
from logging_config import setup_logging
from data_loader import load_dataset, load_split_benchmark
from evaluation.reporting import range_label
from perturbation.range_perturbation import generate_for_range

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT    = "data/perturbations.json"
DEFAULT_SAVE_EVERY = 50   # checkpoint interval (samples)
DEFAULT_RANGES: List[Tuple[float, float]] = [
    (0.90, 1.00), (0.80, 0.90), (0.70, 0.80), (0.60, 0.70), (0.50, 0.60),
]


# ── worker (module-level — required for multiprocessing pickling) ─────────────

def _process_sample_worker(args: tuple) -> Dict[str, Any]:
    """
    Generate perturbations for one sample across all similarity ranges.

    Must be a top-level function so ProcessPoolExecutor can pickle it.
    Each worker process re-imports everything it needs independently.
    """
    (sample_idx, original_input, similarity_ranges,
     n_per_range, sim_metric, seed, budget) = args

    # Strip lone surrogates that JSON's UTF-8 encoder rejects
    if isinstance(original_input, str):
        original_input = original_input.encode('utf-8', errors='ignore').decode('utf-8')

    # Ensure src_v3 is on the path in spawned worker processes (needed on
    # Windows / macOS where the default start method is "spawn", not "fork").
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    from evaluation.reporting import range_label as _range_label
    from perturbation.range_perturbation import generate_for_range as _gen

    t0 = time.time()
    ranges_data: Dict[str, List[Dict]] = {}
    warnings: List[str] = []

    for sim_low, sim_high in similarity_ranges:
        label = _range_label(sim_low, sim_high)
        try:
            candidates = _gen(
                original_text=original_input,
                sim_low=sim_low,
                sim_high=sim_high,
                n=n_per_range,
                similarity_metric=sim_metric,
                seed=seed + sample_idx * 1000,
                budget=budget,
            )
            ranges_data[label] = candidates
            if not candidates:
                warnings.append(f"{label}: 0 candidates")
        except Exception as exc:
            ranges_data[label] = []
            warnings.append(f"{label}: error — {exc}")

    return {
        'sample_idx':     sample_idx,
        'original_input': original_input,
        'ranges':         ranges_data,
        # Stats fields (stripped before saving)
        '_elapsed':  round(time.time() - t0, 2),
        '_n_mods':   sum(len(v) for v in ranges_data.values()),
        '_n_filled': sum(1 for v in ranges_data.values() if v),
        '_n_ranges': len(similarity_ranges),
        '_warnings': warnings,
    }


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_resume_file(path: str) -> Optional[Dict[str, Any]]:
    """
    Load an existing perturbations file for --resume.

    Returns None (and logs a warning) if the file is missing, empty, or
    truncated so that callers can safely treat it as "nothing done yet".
    """
    if not os.path.isfile(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        n = len(data.get('samples', []))
        logger.info(f"Loaded resume file: {n} samples already done ({path})")
        return data
    except json.JSONDecodeError as exc:
        logger.warning(
            f"Resume file is truncated/corrupted (line {exc.lineno}, "
            f"col {exc.colno}): {path}\n"
            f"  Attempting partial recovery..."
        )
        return _recover_partial_json(path)


def _recover_partial_json(path: str) -> Optional[Dict[str, Any]]:
    """
    Best-effort recovery of a truncated perturbations JSON file.

    Reads the file line by line and collects every complete
    ``{"sample_idx": ...}`` object it can parse.  Returns a minimal
    resume dict so already-finished samples are not repeated.
    """
    recovered = []
    # Each sample entry starts with two spaces + "{"  (indent=2 from json.dump)
    # and ends before the next such line.  We collect raw text blocks and try
    # to parse them individually.
    try:
        raw = Path(path).read_text(encoding='utf-8', errors='replace')
    except OSError:
        return None

    # Extract complete sample objects via a simple brace-depth scanner
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start is not None:
                fragment = raw[start:i + 1]
                try:
                    obj = json.loads(fragment)
                    if 'sample_idx' in obj and 'ranges' in obj:
                        recovered.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None

    if not recovered:
        logger.warning("  No complete sample objects found — starting from scratch.")
        return None

    logger.info(f"  Recovered {len(recovered)} complete samples from corrupted file.")
    return {'samples': recovered}


def _atomic_save(data: Dict[str, Any], path: str) -> None:
    """
    Write *data* as JSON to *path* atomically via a sibling temp file.

    On POSIX systems os.replace() is atomic, so a crash during writing
    never leaves a half-written file at *path*.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(_sanitize(data), f, ensure_ascii=False, indent=2)
    os.replace(tmp, out)   # atomic on POSIX


def _parse_ranges(raw: List[List[float]]) -> List[Tuple[float, float]]:
    result = []
    for pair in raw:
        lo, hi = sorted(pair)
        result.append((round(lo, 2), round(hi, 2)))
    return result


def _strip_stats(entry: Dict[str, Any]) -> Dict[str, Any]:
    """Remove internal stat fields before saving."""
    return {k: v for k, v in entry.items() if not k.startswith('_')}


def _clean_str(s: str) -> str:
    """Remove lone surrogate code points that JSON's UTF-8 encoder rejects."""
    return s.encode('utf-8', errors='ignore').decode('utf-8')


def _sanitize(obj: Any) -> Any:
    """Recursively strip surrogates from all strings in a JSON-serialisable object."""
    if isinstance(obj, str):
        return _clean_str(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ── core ──────────────────────────────────────────────────────────────────────

def generate_perturbations(
    config: Dict[str, Any],
    limit: Optional[int] = None,
    resume_from: Optional[Dict[str, Any]] = None,
    benchmark=None,
    num_workers: int = 1,
    save_every: int = DEFAULT_SAVE_EVERY,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate perturbations for all samples and return a serialisable dict.

    Parameters
    ----------
    config :
        Loaded config.yml dict.
    limit :
        Process only the first *limit* samples (useful for testing).
    resume_from :
        Previously saved perturbations dict.  Samples already present are
        skipped and merged verbatim into the output.
    benchmark :
        Pre-loaded Benchmark object. When provided, dataset loading is
        skipped (use this when loading from a named split).
    num_workers :
        Number of parallel worker processes.  ``1`` = sequential (default).
        Set to ``os.cpu_count()`` for maximum throughput.
    save_every :
        Write an intermediate checkpoint every this many completed samples.
        ``0`` disables periodic saves (only saves at the end).
    output_path :
        Destination file for periodic checkpoints.  Required when
        ``save_every > 0``.
    """
    exp_cfg = config.get('experiment', {})

    similarity_ranges = _parse_ranges(
        exp_cfg.get('similarity_ranges', [[hi, lo] for lo, hi in DEFAULT_RANGES])
    )
    n_per_range = exp_cfg.get('n_per_range',          5)
    budget      = exp_cfg.get('perturbation_budget', 300)
    sim_metric  = exp_cfg.get('similarity_metric',   'cosine')
    seed        = exp_cfg.get('seed',                  42)

    # [0.90-1.00] must always be present — used as neighborhood source in experiment
    if (0.90, 1.00) not in similarity_ranges:
        similarity_ranges = [(0.90, 1.00)] + similarity_ranges

    if benchmark is None:
        logger.info("Loading dataset...")
        benchmark = load_dataset(config)
    samples = benchmark.samples[:limit] if limit else benchmark.samples
    total   = len(samples)
    logger.info(f"  → {total} samples from '{benchmark.name}'")

    # Build resume index: sample_idx → already-computed entry
    already_done: Dict[int, Dict] = {}
    if resume_from and 'samples' in resume_from:
        for s in resume_from['samples']:
            already_done[s['sample_idx']] = s
        logger.info(
            f"  Resuming: {len(already_done)} already done, "
            f"{total - len(already_done)} remaining."
        )

    # Separate samples into done vs to-do
    todo: List[Tuple[int, Any]] = [
        (idx, sample)
        for idx, sample in enumerate(samples)
        if idx not in already_done
    ]

    completed: List[Dict[str, Any]] = []
    t_global = time.time()

    # Periodic checkpoint: called after every `save_every` newly completed samples
    def _checkpoint(new_entries: List[Dict[str, Any]]) -> None:
        if not (save_every and output_path and new_entries):
            return
        all_so_far = list(already_done.values()) + new_entries
        all_so_far.sort(key=lambda x: x['sample_idx'])
        _atomic_save(
            {
                'metadata': {
                    'dataset':             benchmark.name,
                    'n_samples':           total,
                    'n_per_range':         n_per_range,
                    'similarity_metric':   sim_metric,
                    'perturbation_budget': budget,
                    'seed':                seed,
                    'similarity_ranges':   [[lo, hi] for lo, hi in similarity_ranges],
                    'generated_at':        datetime.now(timezone.utc).isoformat(),
                },
                'samples': all_so_far,
            },
            output_path,
        )
        logger.info(
            f"  [checkpoint] {len(all_so_far)} samples saved → {output_path}"
        )

    if num_workers > 1 and todo:
        completed = _run_parallel(
            todo, similarity_ranges, n_per_range, sim_metric, seed, budget,
            num_workers=num_workers,
            total=total,
            n_done=len(already_done),
            save_every=save_every,
            checkpoint_fn=_checkpoint,
        )
    else:
        completed = _run_sequential(
            todo, similarity_ranges, n_per_range, sim_metric, seed, budget,
            total=total,
            n_done=len(already_done),
            save_every=save_every,
            checkpoint_fn=_checkpoint,
        )

    total_elapsed = time.time() - t_global
    n_processed   = len(todo)
    logger.info(
        f"\nDone — {n_processed} samples processed in {total_elapsed:.1f}s "
        f"({total_elapsed / max(n_processed, 1):.2f}s/sample)"
    )

    # Merge already-done + newly processed, sort by sample_idx
    all_entries = list(already_done.values()) + completed
    all_entries.sort(key=lambda x: x['sample_idx'])

    return {
        'metadata': {
            'dataset':             benchmark.name,
            'n_samples':           total,
            'n_per_range':         n_per_range,
            'similarity_metric':   sim_metric,
            'perturbation_budget': budget,
            'seed':                seed,
            'similarity_ranges':   [[lo, hi] for lo, hi in similarity_ranges],
            'generated_at':        datetime.now(timezone.utc).isoformat(),
        },
        'samples': all_entries,
    }


# ── sequential and parallel runners ──────────────────────────────────────────

def _run_sequential(
    todo: List[Tuple[int, Any]],
    similarity_ranges, n_per_range, sim_metric, seed, budget,
    total: int, n_done: int,
    save_every: int = 0,
    checkpoint_fn=None,
) -> List[Dict[str, Any]]:
    results = []
    for i, (sample_idx, sample) in enumerate(todo, start=1):
        t0 = time.time()
        logger.info(
            f"[{n_done + i}/{total}] "
            f"{sample.input[:70].replace(chr(10), ' ')}..."
        )

        args = (sample_idx, sample.input, similarity_ranges,
                n_per_range, sim_metric, seed, budget)
        entry = _process_sample_worker(args)

        for w in entry['_warnings']:
            logger.warning(f"  {w}")
        logger.info(
            f"  {entry['_elapsed']:.1f}s | "
            f"{entry['_n_mods']} modifications | "
            f"{entry['_n_filled']}/{entry['_n_ranges']} ranges filled"
        )
        results.append(_strip_stats(entry))

        if save_every and checkpoint_fn and (i % save_every == 0):
            checkpoint_fn(results)

    return results


def _run_parallel(
    todo: List[Tuple[int, Any]],
    similarity_ranges, n_per_range, sim_metric, seed, budget,
    num_workers: int, total: int, n_done: int,
    save_every: int = 0,
    checkpoint_fn=None,
) -> List[Dict[str, Any]]:
    logger.info(
        f"Parallel mode: {num_workers} workers, "
        f"{len(todo)} samples to process."
    )

    worker_args = [
        (idx, sample.input, similarity_ranges, n_per_range, sim_metric, seed, budget)
        for idx, sample in todo
    ]

    results: List[Dict[str, Any]] = []
    n_completed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_idx = {
            executor.submit(_process_sample_worker, args): args[0]
            for args in worker_args
        }

        for future in as_completed(future_to_idx):
            sample_idx = future_to_idx[future]
            n_completed += 1

            try:
                entry = future.result()
            except Exception as exc:
                logger.error(f"  Sample {sample_idx} raised an exception: {exc}")
                results.append({
                    'sample_idx':     sample_idx,
                    'original_input': '',
                    'ranges':         {},
                })
                continue

            for w in entry['_warnings']:
                logger.warning(f"  sample {sample_idx} — {w}")

            logger.info(
                f"  [{n_done + n_completed}/{total}] "
                f"sample {sample_idx} | "
                f"{entry['_elapsed']:.1f}s | "
                f"{entry['_n_mods']} mods | "
                f"{entry['_n_filled']}/{entry['_n_ranges']} ranges filled"
            )
            results.append(_strip_stats(entry))

            if save_every and checkpoint_fn and (n_completed % save_every == 0):
                # Sort before checkpointing so the file is always consistent
                results.sort(key=lambda x: x['sample_idx'])
                checkpoint_fn(results)

    # Restore sample order (as_completed delivers in completion order)
    results.sort(key=lambda x: x['sample_idx'])
    return results


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Pre-generate perturbations for the similarity experiment'
    )
    parser.add_argument('--config',    type=str, default='config.yml',
                        help='Path to config.yml')
    parser.add_argument('--output',    type=str, default=None,
                        help='Output JSON path (default: data/perturbations.json)')
    parser.add_argument('--limit',     type=int, default=None,
                        help='Process only the first N samples')
    parser.add_argument('--resume',    action='store_true',
                        help='Resume from existing output file (skip already-done samples). '
                             'Handles truncated/corrupted files via partial recovery.')
    parser.add_argument(
        '--save-every', type=int, default=DEFAULT_SAVE_EVERY, metavar='N',
        help=f'Save a checkpoint every N completed samples (default: {DEFAULT_SAVE_EVERY}). '
             '0 = only save at the end (risky for long runs).',
    )
    parser.add_argument(
        '--dataset-split', type=str, default=None,
        choices=['leak', 'unleak'],
        help=(
            'Load from datasets.leak or datasets.unleak instead of the '
            'top-level dataset key.  Use this to generate separate '
            'perturbation files for members and non-members.'
        ),
    )
    parser.add_argument(
        '--workers', type=int, default=1,
        metavar='N',
        help=(
            'Number of parallel worker processes (default: 1 = sequential). '
            'Each worker handles one sample at a time across all ranges. '
            'Good values: 4–16 on a multi-core CPU. '
            'Results are saved once at the end; use --resume for crash recovery.'
        ),
    )
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    parser.add_argument('--log-file',  type=str, default=None)
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level), log_file=args.log_file)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = str(_SRC_DIR / config_path)
    config = load_config(config_path)

    output_path = args.output or str(_SRC_DIR / DEFAULT_OUTPUT)

    # Load existing file if resuming (handles truncated/corrupted files)
    resume_data: Optional[Dict] = None
    if args.resume:
        resume_data = _load_resume_file(output_path)

    # Load benchmark from named split if requested
    benchmark = None
    if args.dataset_split:
        benchmark = load_split_benchmark(config, args.dataset_split)
        logger.info(f"Using split '{args.dataset_split}': {len(benchmark.samples)} samples")

    num_workers = max(1, args.workers)
    if num_workers > 1:
        logger.info(f"Using {num_workers} parallel worker processes.")

    result = generate_perturbations(
        config,
        limit=args.limit,
        resume_from=resume_data,
        benchmark=benchmark,
        num_workers=num_workers,
        save_every=args.save_every,
        output_path=output_path,
    )

    # Final atomic write
    _atomic_save(result, output_path)

    logger.info(f"Saved to: {output_path}")
    meta = result['metadata']
    print(f"\nPerturbations saved to: {output_path}")
    print(f"Samples   : {meta['n_samples']}")
    print(f"Ranges    : {[str(r) for r in meta['similarity_ranges']]}")
    print(f"Generated : {meta['generated_at']}")
    print(f"\nTo run the experiment:")
    print(f"  python experiment.py --config {args.config} "
          f"--perturbations-file {output_path}")


if __name__ == '__main__':
    main()
