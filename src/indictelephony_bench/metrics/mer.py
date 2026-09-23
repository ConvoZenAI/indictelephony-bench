"""Mixed error rate (MER) for code-switched speech.

WER and CER each misjudge one half of a code-mixed utterance. WER treats an
agglutinated Kannada or Malayalam token as one indivisible word, so a single
wrong suffix costs as much as a missing English noun. CER goes the other way and
lets a mangled English word cost only a couple of characters.

MER scores each script in its natural unit, the same construction as the MER used
for Mandarin-English code-switching (Mandarin characters, English words):

  * a token containing Latin letters or digits is one unit (an English word, or
    a digit after number canonicalisation);
  * a native-script token is split into aksharas - orthographic syllables: a base
    character with its vowel signs and other combining marks, and any consonant
    joined to it through a virama.

Both texts go through the scoring normalizer first, exactly as for WER and CER,
and edits are counted over the unit sequences with the same Levenshtein
alignment. For a purely English utterance MER equals WER.
"""
from __future__ import annotations

import unicodedata

from jiwer import process_words

from indictelephony_bench.normalize import normalize

_JOINERS = {"‌", "‍"}  # ZWNJ, ZWJ


def _is_virama(char: str) -> bool:
    return unicodedata.combining(char) == 9


def _is_mark(char: str) -> bool:
    return unicodedata.category(char) in {"Mn", "Mc", "Me"}


def aksharas(token: str) -> list[str]:
    """Split one native-script token into orthographic syllables."""
    units: list[str] = []
    for char in token:
        attach = units and (
            _is_mark(char)
            or char in _JOINERS
            or _is_virama(units[-1][-1])
            or units[-1][-1] in _JOINERS
        )
        if attach:
            units[-1] += char
        else:
            units.append(char)
    return units


def _is_latin_unit(token: str) -> bool:
    return any(("a" <= c <= "z") or c.isdigit() for c in token)


def mixed_units(normalized_text: str) -> list[str]:
    """Split normalized text into mixed units: Latin tokens whole, Indic by akshara.

    This is what stops one script dominating the error count in code-mixed text.
    """
    units: list[str] = []
    for token in normalized_text.split():
        if _is_latin_unit(token):
            units.append(token)
        else:
            units.extend(aksharas(token))
    return units


def compute_mer_counts(ground_truth: str, prediction: str) -> tuple[float, int, int]:
    """Return (row_mer, edit_count, reference_units) over normalized text.

    A reference that normalizes to nothing contributes neither edits nor length,
    matching compute_wer_counts.
    """
    ref_units = mixed_units(normalize(ground_truth or ""))
    hyp_units = mixed_units(normalize(prediction or ""))
    if not ref_units:
        return (0.0 if not hyp_units else 1.0), 0, 0
    if not hyp_units:
        return 1.0, len(ref_units), len(ref_units)
    # Units never contain spaces, so joining with spaces and aligning as words
    # aligns exactly the unit sequences.
    output = process_words(" ".join(ref_units), " ".join(hyp_units))
    edits = output.substitutions + output.deletions + output.insertions
    reference_length = output.substitutions + output.deletions + output.hits
    return edits / reference_length, int(edits), int(reference_length)
