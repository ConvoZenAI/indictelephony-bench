"""The five metrics, each corpus-level: total edits over total reference units.

A corpus rate is never the mean of per-utterance rates. Averaging rows lets a
one-word backchannel weigh as much as a twenty-word sentence, and this corpus is
30% backchannels, so the two aggregations are not close.

Each `compute_*_counts` normalizes both sides and returns
`(row_rate, edits, reference_units)`. Sum the last two columns to get a corpus
rate; the row rate is there for inspecting single utterances, not for averaging.
"""
from indictelephony_bench.metrics.cer import (
    char_edit_counts,
    compute_cer,
    compute_cer_counts,
)
from indictelephony_bench.metrics.keywords import (
    compute_keyword_counts,
    compute_keyword_wer,
)
from indictelephony_bench.metrics.mer import aksharas, compute_mer_counts, mixed_units
from indictelephony_bench.metrics.wer import (
    compute_wer,
    compute_wer_counts,
    word_edit_counts,
)

__all__ = [
    "compute_wer_counts", "compute_wer", "word_edit_counts",
    "compute_cer_counts", "compute_cer", "char_edit_counts",
    "compute_mer_counts", "mixed_units", "aksharas",
    "compute_keyword_counts", "compute_keyword_wer",
]
