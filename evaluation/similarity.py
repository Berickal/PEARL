"""
Similarity metrics for comparing text pairs.

All functions return a float in [0, 1] where 1 means identical.
"""
import math
import re
import logging
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bow_cosine(text1: str, text2: str) -> float:
    """Pure-Python bag-of-words cosine similarity (TF only, no IDF)."""
    def _counts(text):
        counts: dict = {}
        for tok in re.findall(r'\b\w+\b', text.lower()):
            counts[tok] = counts.get(tok, 0) + 1
        return counts

    c1, c2 = _counts(text1), _counts(text2)
    vocab = set(c1) | set(c2)
    dot = sum(c1.get(t, 0) * c2.get(t, 0) for t in vocab)
    mag1 = math.sqrt(sum(v * v for v in c1.values()))
    mag2 = math.sqrt(sum(v * v for v in c2.values()))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def _levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1, 1):
        curr = [i]
        for j, c2 in enumerate(s2, 1):
            curr.append(min(prev[j] + 1, curr[-1] + 1, prev[j - 1] + (c1 != c2)))
        prev = curr
    return prev[-1]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate_cosine_similarity(text1: str, text2: str) -> float:
    """
    Character n-gram TF-IDF cosine similarity between two texts.

    Uses sklearn TfidfVectorizer with char_wb analyzer and bigram–4-gram features.
    This captures character-level differences (case, whitespace, swaps) as well as
    word-level differences, giving a meaningful similarity score across all
    perturbation types used in the experiment.

    Falls back to bag-of-words cosine when sklearn is unavailable.
    """
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

        vectorizer = TfidfVectorizer(
            analyzer='char_wb',
            ngram_range=(2, 4),
            lowercase=False,   # preserve case so case-flip perturbations are visible
        )
        tfidf = vectorizer.fit_transform([text1, text2])
        score = float(sk_cosine(tfidf[0:1], tfidf[1:2])[0][0])
        return max(0.0, min(1.0, score))
    except Exception:
        return _bow_cosine(text1, text2)


def calculate_levenshtein_similarity(text1: str, text2: str) -> float:
    """Normalised Levenshtein similarity (character-level)."""
    if not text1 and not text2:
        return 1.0
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    dist = _levenshtein_distance(text1, text2)
    return 1.0 - dist / max(len(text1), len(text2))


def calculate_bleu_similarity(text1: str, text2: str) -> float:
    """
    Sentence-BLEU using text1 as reference and text2 as hypothesis.
    Requires nltk; falls back to 0.0 if unavailable.
    """
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        ref = text1.lower().split()
        hyp = text2.lower().split()
        if not ref or not hyp:
            return 0.0
        smoothie = SmoothingFunction().method4
        score = sentence_bleu([ref], hyp, smoothing_function=smoothie)
        return max(0.0, min(1.0, float(score)))
    except ImportError:
        logger.warning("nltk not available for BLEU; returning 0.0")
        return 0.0
    except Exception as e:
        logger.warning(f"BLEU computation failed: {e}")
        return 0.0


def calculate_codebleu_similarity(text1: str, text2: str) -> float:
    """
    CodeBLEU similarity for code snippets.
    Uses the codebleu package when available; falls back to BLEU.
    """
    if not text1 or not text2:
        return 0.0
    if text1 == text2:
        return 1.0
    try:
        from codebleu import calc_codebleu
        result = calc_codebleu(
            references=[[text1]],
            predictions=[text2],
            lang='python',
        )
        return max(0.0, min(1.0, float(result['codebleu'])))
    except ImportError:
        logger.debug("codebleu not available; falling back to BLEU for code similarity")
        return calculate_bleu_similarity(text1, text2)
    except Exception as e:
        logger.warning(f"CodeBLEU computation failed: {e}; falling back to BLEU")
        return calculate_bleu_similarity(text1, text2)


def calculate_similarity(text1: str, text2: str, metric: str) -> float:
    """Dispatch to the right similarity function by metric name."""
    m = metric.lower()
    if m == 'cosine':
        return calculate_cosine_similarity(text1, text2)
    elif m == 'levenshtein':
        return calculate_levenshtein_similarity(text1, text2)
    elif m == 'bleu':
        return calculate_bleu_similarity(text1, text2)
    elif m == 'codebleu':
        return calculate_codebleu_similarity(text1, text2)
    else:
        raise ValueError(f"Unsupported similarity metric: {metric}")
