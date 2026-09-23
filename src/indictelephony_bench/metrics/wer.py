"""Word error rate.

Reference and hypothesis pass through `normalize.normalize` first, the same
function on both sides, so a difference of writing convention is not charged as
a recognition error. `word_edit_counts` takes strings that are ALREADY
normalized; `compute_wer` normalizes for you.
"""
from jiwer import process_words

from indictelephony_bench.normalize import normalize


def word_edit_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edits, reference_length) between two ALREADY-normalized strings.

    Shared with semantic WER so both metrics count edits the same way.
    """
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0, 0
    if not hyp_words:
        return len(ref_words), len(ref_words)
    output = process_words(reference, hypothesis)
    edits = output.substitutions + output.deletions + output.insertions
    reference_length = output.substitutions + output.deletions + output.hits
    return int(edits), int(reference_length)


def compute_wer_counts(
    ground_truth: str,
    prediction: str,
    language: str | None = None,
) -> tuple[float, int, int]:
    """Return (row_wer, edit_count, reference_length) over normalized text.

    Edit counts are returned alongside the rate because corpus-level WER -
    total edits / total reference words, the standard ASR aggregation - cannot
    be recovered from a column of per-row rates. A mean of rates weights a
    3-word utterance the same as a 30-word one.

    `language` is accepted for call-site compatibility only: the scoring
    normalizer is language-independent.
    """
    gt = normalize(ground_truth or "")
    pred = normalize(prediction or "")

    if not gt.split():
        # Nothing to divide by. Contribute *neither* edits nor reference length,
        # so the row stays out of the corpus figure entirely - counting its
        # insertions against a zero denominator would inflate corpus WER.
        # Reachable because a reference made only of punctuation normalizes to "".
        return (0.0 if not pred.split() else 1.0), 0, 0

    edits, reference_length = word_edit_counts(gt, pred)
    rate = edits / reference_length if reference_length else 0.0
    return float(rate), edits, reference_length


def compute_wer(ground_truth: str, prediction: str, language: str | None = None) -> float:
    """Row-level WER. For a corpus rate, sum `compute_wer_counts` instead."""
    return compute_wer_counts(ground_truth, prediction, language)[0]
