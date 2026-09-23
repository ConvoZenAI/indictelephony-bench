"""Number pairs in semantic WER.

English number words written in an Indian script are read back as English and
must equal the digit side exactly. Native number words of Bengali, Gujarati,
Kannada, Malayalam, Tamil and Telugu are not verified and so never forgiven:
their composition is agglutinative and colloquial, and the curated transcript is
used as written. Cases come from the semantic-WER smoke test.
"""
import pytest

from indictelephony_bench.metrics.semantic import verify_pair
from indictelephony_bench.normalize import normalize


def check(words, digits):
    return verify_pair(normalize(words), normalize(digits), "number", min_similarity=60)[0]


ENGLISH_IN_INDIAN_SCRIPT = [
    ("ಟೂ ಹಂಡ್ರೆಡ್", "2 0 0"),                                  # kn two hundred
    ("વન ફાઇવ સેવન થ્રી", "1 5 7 3"),                           # gu dictated
    ("వన్ థౌసండ్ త్రీ హండ్రెడ్ అండ్ ఫార్టీ టూ", "1 3 4 2"),        # te with "and"
    ("జీరో పాయింట్ జీరో సెవెన్", "0 0 7"),                        # te decimal
    ("టెన్ పాయింట్ నైన్ నైన్", "1 0 9 9"),                         # te decimal
    ("টু থাউজ্যান্ড টোয়েন্টি এইট", "2 0 2 8"),                    # bn year
    ("ফাইভ হান্ড্রেড", "5 0 0"),                                 # bn
    ("ട്വന്റി", "2 0"),                                           # ml
    ("ಒನ್ ಆರ್ ಟೂ", "1 or 2"),                                     # kn "one or two"
    ("એઈટી એઈટી થર્ટી ફાઇવ રુપીસ", "8 0 8 0 3 5 rupees"),        # gu with rupees
    ("सिक्सटी सिक्स", "6 6"),                                    # mr
]


@pytest.mark.parametrize("words,digits", ENGLISH_IN_INDIAN_SCRIPT)
def test_english_number_words_in_indian_script_are_verified(words, digits):
    assert check(words, digits)


def test_wrong_value_is_still_rejected():
    # the judge split "eighty three": "eighty" alone is not 8
    assert not check("এট্টি", "8")


@pytest.mark.parametrize("words,digits", [
    ("নেওয়া", "9 0"),                    # a native word that is not a number
    ("નેવું", "9 0"),                     # gu native 90: not verified by design
    ("দুশো আটচল্লিশ", "2 4 8"),          # bn native 248: not verified by design
    ("ಇಪ್ಪತೈದ್", "2 5"),                 # kn native 25: not verified by design
])
def test_native_number_words_are_not_forgiven(words, digits):
    assert not check(words, digits)


@pytest.mark.parametrize("words,digits", [
    ("টুয়েন্টি ফাইভ থাউজেন্ডের", "2 5 0 0 0"),   # case suffix: deliberately not read
    ("half", "0 5"),
    ("age", "8"),
    ("leven", "1 1"),
])
def test_ambiguous_forms_are_rejected(words, digits):
    assert not check(words, digits)


def test_mixed_native_and_english_tokens_rejected():
    # every token must be an English number word; one native word fails the span
    assert not check("ಟೂ ನೂರ", "2 0 0")


def test_hindi_spelling_variant_merged_by_the_normalizer():
    assert normalize("पच्चास") == normalize("50") == normalize("पचास")
