from indictelephony_bench.metrics.mer import aksharas, compute_mer_counts, mixed_units
from indictelephony_bench.metrics.wer import compute_wer_counts


def test_aksharas_keep_vowel_signs_and_conjuncts():
    assert aksharas("क्या") == ["क्या"]
    assert aksharas("कितने") == ["कि", "त", "ने"]
    assert aksharas("ಕನ್ನಡ") == ["ಕ", "ನ್ನ", "ಡ"]


def test_latin_tokens_stay_words_and_native_tokens_split():
    assert mixed_units("email के regard") == ["email", "के", "regard"]
    assert mixed_units("dinam 24") == ["dinam", "24"]


def test_english_only_mer_equals_wer():
    ref, hyp = "please check my account balance", "please check account balances"
    assert compute_mer_counts(ref, hyp)[1:] == compute_wer_counts(ref, hyp)[1:]


def test_wrong_suffix_costs_one_akshara_not_one_word():
    ref, hyp = "ಮನೆಯಲ್ಲಿ ಇದೆ", "ಮನೆಯಲ್ಲ ಇದೆ"
    _, wer_edits, _ = compute_wer_counts(ref, hyp)
    _, mer_edits, mer_len = compute_mer_counts(ref, hyp)
    assert wer_edits == 1
    assert mer_edits == 1 and mer_len == len(mixed_units(ref))


def test_empty_reference_and_empty_hypothesis():
    assert compute_mer_counts("", "") == (0.0, 0, 0)
    rate, edits, length = compute_mer_counts("email के", "")
    assert rate == 1.0 and edits == length == 2
