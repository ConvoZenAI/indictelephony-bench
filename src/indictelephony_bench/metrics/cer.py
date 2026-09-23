"""Character error rate.

CER carries more information than WER for Malayalam, Kannada and Telugu, where
whitespace is a poor word boundary and one agglutinated token counts as a whole
word error under WER. The CER/WER ratio is worth reading on its own: a low ratio
means a wrong word is only slightly wrong, usually an inflected ending.
"""
from jiwer import process_characters

from indictelephony_bench.normalize import normalize


def char_edit_counts(reference: str, hypothesis: str) -> tuple[int, int]:
    """(edits, reference_length) in characters between ALREADY-normalized strings.

    A true Levenshtein distance. Scorers that sum difflib opcode spans instead
    are computing an approximation and will differ slightly, so do not mix the
    two in one table.
    """
    if not reference:
        return 0, 0
    if not hypothesis:
        return len(reference), len(reference)
    output = process_characters(reference, hypothesis)
    edits = output.substitutions + output.deletions + output.insertions
    reference_length = output.substitutions + output.deletions + output.hits
    return int(edits), int(reference_length)


def compute_cer_counts(
    ground_truth: str,
    prediction: str,
    language: str | None = None,
) -> tuple[float, int, int]:
    """Return (row_cer, edit_count, reference_length) over normalized text.

    CER matters more than WER for Malayalam/Kannada/Telugu, where whitespace is
    a poor word boundary and a single agglutinated token counts as one whole
    word error under WER. Uses the same scoring normalizer as WER.
    """
    gt = normalize(ground_truth or "")
    pred = normalize(prediction or "")

    if not gt:
        # Contribute neither edits nor reference length - see compute_wer_counts.
        return (0.0 if not pred else 1.0), 0, 0

    edits, reference_length = char_edit_counts(gt, pred)
    rate = edits / reference_length if reference_length else 0.0
    return float(rate), edits, reference_length


def compute_cer(ground_truth: str, prediction: str, language: str | None = None) -> float:
    """Row-level CER. For a corpus rate, sum `compute_cer_counts` instead."""
    return compute_cer_counts(ground_truth, prediction, language)[0]
