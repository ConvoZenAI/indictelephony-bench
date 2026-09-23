"""Numbers said one way and written another must score as the same thing.

Each case is a pair a provider and a curator could plausibly produce for the
same audio. They must normalize identically. The counter-cases guard the fix:
a dictated digit run is not a cardinal, and "point" is not always a decimal.
"""
import pytest

from indictelephony_bench.normalize import normalize

SAME = [
    # digit + scale word, the form Deepgram/Saaras/Scribe actually emit
    ("eight lakhs", "8 lakhs"),
    ("eight lakh", "8 lakh"),
    ("twenty five lakh", "25 lakh"),
    ("two thousand", "2 thousand"),
    ("twenty thousand", "20 thousand"),
    ("five hundred", "5 hundred"),
    ("five hundred", "500"),
    ("three crore", "3 crore"),
    ("one lakh fifteen thousand", "1 लाख 15 हजार"),
    # Indic scale words
    ("पंद्रह हजार", "15 हजार"),
    ("एक लाख", "1 लाख"),
    ("दो हजार", "2 हजार"),
    # decimals: the written dot is punctuation, the spoken joiner is "point"
    ("one point two five", "1.25"),
    ("eight point seven five percent", "8.75%"),
    # symbols the curator writes as words
    ("ninety percent", "90%"),
    ("eighteen percent gst", "18% GST"),
    ("rupees five hundred", "₹500"),
    # ordinals and times
    ("twenty eighth may", "28th may"),
    ("two thirty", "2:30"),
]

DIFFERENT = [
    # "point" outside a number is an ordinary word, not a decimal joiner
    ("the point is clear", "the 0 is clear"),
]


@pytest.mark.parametrize("spoken,written", SAME)
def test_same_number_normalizes_identically(spoken, written):
    assert normalize(spoken) == normalize(written)


@pytest.mark.parametrize("a,b", DIFFERENT)
def test_distinct_readings_stay_distinct(a, b):
    assert normalize(a) != normalize(b)


def test_dictated_run_survives_as_a_sequence():
    assert normalize("my number is 9 8 7 6 5") == "my number is 9 8 7 6 5"
    assert normalize("nine two nine three") == "9 2 9 3"


def test_composed_and_dictated_share_one_digit_form():
    """Deliberate: one digit per token means 9293 and a dictated 9 2 9 3 agree.

    Scoring cares that the digits are right, not how the speaker grouped them,
    and this way one wrong digit costs exactly one token either way.
    """
    assert normalize("nine thousand two hundred ninety three") == normalize("nine two nine three")


def test_one_second_is_a_duration_not_an_ordinal():
    assert normalize("wait one second") == "wait 1 second"


def test_adjacent_literals_are_not_composed():
    # "2 3" is two numbers; only a scale word may continue a literal
    assert normalize("2 3") == "2 3"
