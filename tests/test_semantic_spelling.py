"""The spelling check must not forgive two different English words.

Consonant-only matching is right for Indic scripts, where the same word is
written with different vowel diacritics, and wrong for English, where the vowel
carries the distinction. No similarity threshold separates the two groups
("insurance"/"assurance" scores 82, "grey"/"gray" scores 75), so a Latin pair is
never guessed at: the English variants worth forgiving are enumerated in the
scoring normalizer and merged before the judge ever sees the row.
"""
import pytest

from indictelephony_bench.metrics.semantic import verify_pair
from indictelephony_bench.normalize import normalize

DIFFERENT_ENGLISH_WORDS = [
    ("insurance", "assurance"), ("loan", "lane"), ("bill", "bull"),
    ("month", "mouth"), ("card", "cord"), ("madam", "modem"),
    ("policy", "police"), ("call", "cool"), ("data", "date"), ("rate", "rite"),
]

NORMALIZER_HANDLES = [("colour", "color"), ("okay", "ok"), ("maam", "mam")]


@pytest.mark.parametrize("hyp,ref", DIFFERENT_ENGLISH_WORDS)
def test_latin_spelling_pair_is_never_forgiven(hyp, ref):
    accepted, _ = verify_pair(hyp, ref, "spelling", min_similarity=60)
    assert not accepted


@pytest.mark.parametrize("a,b", NORMALIZER_HANDLES)
def test_real_english_variants_are_merged_before_the_judge(a, b):
    # These never reach verify_pair, so rejecting Latin spelling costs nothing.
    assert normalize(a) == normalize(b)


def test_indic_spelling_variant_still_accepted():
    accepted, _ = verify_pair("ठिक", "ठीक", "spelling", min_similarity=60)
    assert accepted


@pytest.mark.parametrize("hyp,ref", [("कार", "car"), ("लोन", "loan")])
def test_cross_script_pair_still_accepted(hyp, ref):
    accepted, _ = verify_pair(hyp, ref, "script", min_similarity=60)
    assert accepted


def test_wrong_number_still_rejected():
    accepted, reason = verify_pair("9 0 0 0 0 0", "8 0 0 0 0 0", "number", min_similarity=60)
    assert not accepted and "digits on exactly one side" in reason


def test_negation_is_not_a_filler():
    accepted, reason = verify_pair("", "नहीं", "filler", min_similarity=60)
    assert not accepted and "hesitation" in reason
