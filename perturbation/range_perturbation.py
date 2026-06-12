"""
Range-targeted perturbation.

Generates modified versions of a text whose cosine similarity to the original
falls within a specified range [sim_low, sim_high].

Strategy per range
------------------
[0.90-1.00]  Very light surface noise: whitespace tweaks, mild case flips.
[0.80-0.90]  Char-level noise: case flips, adjacent-char swaps.
[0.70-0.80]  Word-level + char-level: moderate swaps, light synonyms.
[0.60-0.70]  Word substitution: synonyms, spelling replacements.
[0.50-0.60]  Aggressive substitution: high-prob synonyms, token replacement.

When a single perturbation type cannot supply enough candidates in the target
range (common for the extreme ranges), types are mixed — every generated
candidate is scored and accepted only if its similarity falls in [sim_low, sim_high].
"""
import logging
import random
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Strategy table
# Each entry: (perturbation_type, constructor_kwargs)
# Listed roughly from lightest to heaviest within each range.
# ---------------------------------------------------------------------------
_RANGE_STRATEGIES: Dict[Tuple[float, float], List[Tuple[str, Dict[str, Any]]]] = {
    # Calibrated against char-ngram TF-IDF cosine on ~90-char English text.
    # Each strategy is listed with typical avg similarity in parentheses.

    (0.90, 1.00): [
        # whitespace (tiny) → 0.93 avg; swap (very small) → 0.96 avg
        ('whitespace',  {'remove_prob': 0.05, 'add_prob': 0.02}),   # ~0.93
        ('swap_chars',  {'prob': 0.01}),                              # ~0.97
        ('whitespace',  {'remove_prob': 0.08, 'add_prob': 0.03}),   # ~0.91
        ('swap_chars',  {'prob': 0.02}),                              # ~0.94
        ('change_case', {'prob': 0.03}),                              # ~0.93
    ],
    (0.80, 0.90): [
        # swap (small) → 0.885 avg; case (small) → 0.847 avg
        ('swap_chars',  {'prob': 0.03}),                              # ~0.89
        ('change_case', {'prob': 0.05}),                              # ~0.85
        ('swap_chars',  {'prob': 0.04}),                              # ~0.87
        ('change_case', {'prob': 0.04}),                              # ~0.87
        ('synonyms',    {'prob': 0.10}),                              # ~0.85 (if nlpaug)
    ],
    (0.70, 0.80): [
        # whitespace (moderate) → 0.795 avg
        ('whitespace',  {'remove_prob': 0.15, 'add_prob': 0.08}),   # ~0.80
        ('change_case', {'prob': 0.07}),                              # ~0.77
        ('whitespace',  {'remove_prob': 0.18, 'add_prob': 0.10}),   # ~0.76
        ('change_case', {'prob': 0.06}),                              # ~0.78
        ('synonyms',    {'prob': 0.20}),                              # ~0.75 (if nlpaug)
    ],
    (0.60, 0.70): [
        # whitespace (heavy) → 0.660 avg; swap (moderate) → 0.658 avg
        ('whitespace',  {'remove_prob': 0.30, 'add_prob': 0.15}),   # ~0.66
        ('swap_chars',  {'prob': 0.08}),                              # ~0.66
        ('whitespace',  {'remove_prob': 0.25, 'add_prob': 0.12}),   # ~0.69
        ('swap_chars',  {'prob': 0.07}),                              # ~0.68
        ('synonyms',    {'prob': 0.35}),                              # ~0.65 (if nlpaug)
        ('token_replacement', {'replacement_prob': 0.20}),            # (if nlpaug)
    ],
    (0.50, 0.60): [
        # case (high) → 0.532 avg; swap (high) → 0.510 avg
        ('change_case', {'prob': 0.20}),                              # ~0.53
        ('swap_chars',  {'prob': 0.15}),                              # ~0.51
        ('change_case', {'prob': 0.18}),                              # ~0.55
        ('swap_chars',  {'prob': 0.12}),                              # ~0.54
        ('synonyms',    {'prob': 0.50}),                              # (if nlpaug)
        ('token_replacement', {'replacement_prob': 0.35}),            # (if nlpaug)
    ],
}

# Normalised keys (low, high) for look-up
_KNOWN_RANGES = list(_RANGE_STRATEGIES.keys())

# Transformer types backed by nlpaug — expensive to init, have internal randomness,
# so they benefit from caching.  Simple types (whitespace, swap_chars, change_case)
# must be recreated each attempt to get a different output.
_NLPAUG_TYPES = frozenset({'synonyms', 'synonym', 'token_replacement', 'token'})


def _find_strategies(sim_low: float, sim_high: float) -> List[Tuple[str, Dict[str, Any]]]:
    """Return the strategy list for the given range, or a sensible default."""
    key = (round(sim_low, 2), round(sim_high, 2))
    if key in _RANGE_STRATEGIES:
        return _RANGE_STRATEGIES[key]
    # Fallback: pick the closest range
    best = min(_KNOWN_RANGES, key=lambda k: abs(k[0] - sim_low) + abs(k[1] - sim_high))
    logger.warning(f"No exact strategy for range ({sim_low}, {sim_high}); using {best}")
    return _RANGE_STRATEGIES[best]


def _make_transformer(perturbation_type: str, kwargs: Dict[str, Any], seed: int):
    """Instantiate a perturbation transformer."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    t = perturbation_type.lower()
    if t == 'whitespace':
        from perturbation.whitespace_perturbation import WhitespacePerturbation
        return WhitespacePerturbation(seed=seed, max_outputs=1, **kwargs)
    elif t == 'change_case':
        from perturbation.change_char_case import ChangeCharCase
        return ChangeCharCase(seed=seed, max_outputs=1, **kwargs)
    elif t == 'swap_chars':
        from perturbation.swap_characters import SwapCharacters
        return SwapCharacters(seed=seed, **kwargs)
    elif t in ('synonyms', 'synonym'):
        from perturbation.synonym_substitution import SynonymSubstitution
        prob = kwargs.get('prob', 0.3)
        return SynonymSubstitution(seed=seed, prob=prob, max_outputs=1)
    elif t in ('token_replacement', 'token'):
        from perturbation.token_replacement import TokenReplacement
        replacement_prob = kwargs.get('replacement_prob', 0.2)
        return TokenReplacement(seed=seed, max_outputs=1, replacement_prob=replacement_prob)
    else:
        raise ValueError(f"Unknown perturbation type: {perturbation_type}")


def generate_for_range(
    original_text: str,
    sim_low: float,
    sim_high: float,
    n: int,
    similarity_metric: str = 'cosine',
    seed: int = 42,
    budget: int = 300,
) -> List[Dict[str, Any]]:
    """
    Generate up to *n* modified texts whose similarity to *original_text*
    falls within [sim_low, sim_high].

    Parameters
    ----------
    original_text : str
    sim_low, sim_high : float
        Target similarity interval (inclusive on both ends).
    n : int
        Desired number of accepted candidates.
    similarity_metric : str
        One of 'cosine', 'levenshtein', 'bleu'.
    seed : int
        Base random seed.
    budget : int
        Maximum total generation attempts across all strategies.

    Returns
    -------
    List[Dict]  — each dict has keys:
        modified_text, similarity, perturbation_type
    """
    from evaluation.similarity import calculate_similarity

    strategies = _find_strategies(sim_low, sim_high)
    accepted: List[Dict[str, Any]] = []
    seen_texts = {original_text}
    attempts = 0
    rng = random.Random(seed)

    # Cache only nlpaug-backed transformers (expensive to init, have internal
    # randomness so repeated calls produce different outputs).
    # Simple transformers (whitespace, swap_chars, change_case) are re-created
    # each attempt so that a different seed produces a different perturbation.
    _nlpaug_cache: Dict[tuple, Any] = {}

    def _get_transformer(strat_type: str, strat_kwargs: Dict[str, Any], attempt_seed: int):
        t = strat_type.lower()
        if t in _NLPAUG_TYPES:
            cache_key = (t, tuple(sorted(strat_kwargs.items())))
            if cache_key not in _nlpaug_cache:
                _nlpaug_cache[cache_key] = _make_transformer(strat_type, strat_kwargs, attempt_seed)
            return _nlpaug_cache[cache_key]
        # Lightweight — new instance per attempt for output diversity
        return _make_transformer(strat_type, strat_kwargs, attempt_seed)

    # Cycle through strategies until budget exhausted or n reached
    strategy_cycle = strategies * (budget // max(len(strategies), 1) + 1)

    for attempt_idx in range(budget):
        if len(accepted) >= n:
            break

        strat_type, strat_kwargs = strategy_cycle[attempt_idx % len(strategies)]
        attempt_seed = seed + attempt_idx + rng.randint(0, 10_000)

        try:
            transformer = _get_transformer(strat_type, strat_kwargs, attempt_seed)
            candidates = transformer.generate(original_text)
        except Exception as e:
            logger.debug(f"Transformer {strat_type} failed (seed={attempt_seed}): {e}")
            attempts += 1
            continue

        for candidate in candidates:
            if candidate in seen_texts:
                continue
            seen_texts.add(candidate)
            attempts += 1

            sim = calculate_similarity(original_text, candidate, similarity_metric)
            if sim_low <= sim <= sim_high:
                accepted.append({
                    'modified_text': candidate,
                    'similarity': round(sim, 6),
                    'perturbation_type': strat_type,
                })
                if len(accepted) >= n:
                    break

    if len(accepted) < n:
        logger.warning(
            f"Range ({sim_low:.2f}, {sim_high:.2f}): wanted {n}, "
            f"got {len(accepted)} after {attempts} attempts."
        )

    return accepted
