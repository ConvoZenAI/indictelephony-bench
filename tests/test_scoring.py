"""Offline tests for the scoring path: normalizer, WER/CER counts, keywords,
semantic-WER verification. No network, no audio."""
import math
import unittest
import warnings
from pathlib import Path

import pandas as pd
import pytest

from indictelephony_bench.data import align, load_predictions
from indictelephony_bench.metrics.cer import compute_cer_counts
from indictelephony_bench.metrics.keywords import (
    compute_keyword_counts,
    compute_keyword_wer,
    parse_keywords,
)
from indictelephony_bench.scoring import bootstrap_ci

pytest.importorskip("indictelephony_bench.metrics.semantic",
                    reason="semantic WER needs the [semantic] extra")
from indictelephony_bench.metrics.semantic import apply_equivalences, verify_pair
from indictelephony_bench.metrics.wer import compute_wer_counts
from indictelephony_bench.normalize import normalize


class NormalizerTests(unittest.TestCase):
    def test_indic_digits_fold_to_ascii(self):
        self.assertEqual(normalize("फोन ५० है"), normalize("फोन 50 है"))
        self.assertEqual(normalize("௨௪"), normalize("24"))

    def test_zero_width_joiners_do_not_split_or_distinguish_words(self):
        self.assertEqual(normalize("क्‌ष"), normalize("क्ष"))
        self.assertEqual(len(normalize("ab‍cd").split()), 1)

    def test_indic_marks_survive(self):
        self.assertEqual(normalize("क्या हाल है?"), "क्या हाल है")

    def test_number_words_and_digits_agree(self):
        self.assertEqual(normalize("twenty four thousand"), normalize("24,000"))


class WerCerTests(unittest.TestCase):
    def test_empty_prediction_is_full_deletion(self):
        self.assertEqual(compute_wer_counts("one two three words", ""), (1.0, 4, 4))

    def test_empty_reference_contributes_nothing(self):
        rate, edits, length = compute_wer_counts("...", "hello")
        self.assertEqual((edits, length), (0, 0))

    def test_punctuation_and_case_are_free(self):
        self.assertEqual(compute_wer_counts("Okay, sir.", "ok sir")[1], 0)

    def test_cer_counts(self):
        self.assertEqual(compute_cer_counts("abcd", "abxd")[1:], (1, 4))


class KeywordTests(unittest.TestCase):
    def test_parse_handles_newlines_commas_and_duplicates(self):
        self.assertEqual(parse_keywords("loan, EMI\npolicy number\nloan\n"), ["loan", "emi", "policy number"])

    def test_no_keywords_is_nan_not_perfect(self):
        self.assertTrue(math.isnan(compute_keyword_wer("", "anything")))
        self.assertEqual(compute_keyword_counts("  \n", "x"), (0, 0))

    def test_suffixed_indic_form_matches(self):
        self.assertEqual(compute_keyword_counts("website", "அந்த websiteல பாருங்க"), (1, 1))

    def test_no_match_inside_another_word(self):
        self.assertEqual(compute_keyword_counts("car", "the scar is old"), (0, 1))

    def test_multiword_needs_order(self):
        self.assertEqual(compute_keyword_counts("policy number", "number policy"), (0, 1))
        self.assertEqual(compute_keyword_counts("policy number", "your policy numbers"), (1, 1))

    def test_numbers_match_across_forms(self):
        self.assertEqual(compute_keyword_counts("twenty four thousand", "it is 24000 rupees"), (1, 1))


class KeywordScopeTests(unittest.TestCase):
    def test_keywords_absent_from_reference_are_not_scored(self):
        ref = "random ಆಗಿ car ನೋಡ್ಬೇಕು Whitefield ಅಲ್ಲಿ"
        self.assertEqual(compute_keyword_counts("car, white filed", "car ನೋಡಿ", reference=ref), (1, 1))
        self.assertEqual(compute_keyword_counts("white filed", "anything", reference=ref), (0, 0))

    def test_number_keyword_matches_inside_a_longer_number_phrase(self):
        ref = "eighty thousand eight lakhs ಇದು"
        self.assertEqual(compute_keyword_counts("eighty thousand, eight lakhs", ref, reference=ref), (2, 2))

    def test_english_inflection_only(self):
        self.assertEqual(compute_keyword_counts("loan", "two loans"), (1, 1))
        self.assertEqual(compute_keyword_counts("car", "my card"), (0, 1))

    def test_digit_group_comma_is_not_a_separator(self):
        self.assertEqual(parse_keywords("24,000, loan"), [normalize("24000"), "loan"])


class SemanticVerificationTests(unittest.TestCase):
    def test_loanword_in_local_script_is_accepted(self):
        self.assertTrue(verify_pair("ಕಾರ್", "car", "script", 60)[0])
        self.assertTrue(verify_pair("लोन", "loan", "script", 60)[0])

    def test_loanwords_with_non_phonetic_english_spelling_are_accepted(self):
        for hyp, ref in [("சார்", "sir"), ("স্যার", "sir"), ("કોલ", "call"), ("കോൾ", "call"),
                         ("ഗൂഗിൾ", "google"), ("নম্বর", "number"), ("ഡിസൈഡ്", "decide"),
                         ("ജസ്റ്റ്", "just"), ("ഓൺലൈൻ", "online"), ("ફોલો", "follow"), ("மீட்டிங்", "meeting")]:
            self.assertTrue(verify_pair(normalize(hyp), ref, "script", 60)[0], (hyp, ref))

    def test_different_words_are_rejected(self):
        self.assertFalse(verify_pair("ಹೌದು", "no", "script", 60)[0])
        self.assertFalse(verify_pair("yes", "ok", "spelling", 60)[0])
        # audited false acceptances: different Gujarati words, not spelling variants
        for hyp, ref in [("શું", "સુ"), ("કહી", "કઈ"), ("ચુંમાળી", "ચુંબાલીસ")]:
            self.assertFalse(verify_pair(normalize(hyp), normalize(ref), "spelling", 60)[0], (hyp, ref))

    def test_spelling_variants_are_accepted(self):
        for hyp, ref in [("அஞ்சு", "அஞ்சி"), ("எண்பத்தி", "எம்பத்தி"), ("যত", "যতো"), ("இரநூறு", "எரநூறு")]:
            self.assertTrue(verify_pair(normalize(hyp), normalize(ref), "spelling", 60)[0], (hyp, ref))

    def test_number_pairs_need_a_verified_value(self):
        self.assertTrue(verify_pair(normalize("24000"), "twenty four thousand", "number", 60)[0])
        self.assertTrue(verify_pair(normalize("11 am"), "eleven am", "number", 60)[0])
        self.assertFalse(verify_pair(normalize("11"), "leven", "number", 60)[0])       # misrecognition
        self.assertFalse(verify_pair(normalize("12"), "eleven", "number", 60)[0])
        gujarati = (normalize("944"), normalize("નવસો ચુંબાલીસ"))
        self.assertFalse(verify_pair(*gujarati, "number", 60)[0])
        self.assertTrue(verify_pair(*gujarati, "number", 60, allow_unverified_numbers=True)[0])

    def test_currency_symbol_is_its_own_token(self):
        self.assertEqual(normalize("₹287"), normalize("Rs. 287"))

    def test_unknown_type_and_digits_outside_number_are_rejected(self):
        self.assertFalse(verify_pair("5", "6", "synonym", 60)[0])
        self.assertFalse(verify_pair("5", "five", "spelling", 60)[0])

    def test_filler_rules(self):
        self.assertTrue(verify_pair("hmm", "", "filler", 60)[0])
        self.assertTrue(verify_pair("আহ", "", "filler", 60)[0])
        self.assertFalse(verify_pair("haan", "", "filler", 60)[0])   # "haan" means yes
        self.assertFalse(verify_pair("hmm", "ah", "filler", 60)[0])

    def test_semantic_edits_never_exceed_lexical(self):
        ref, hyp = normalize("naan car vaanginen"), normalize("naan ಕಾರ್ vaanginen hmm")
        out = apply_equivalences(ref, hyp, [
            {"hypothesis_span": "ಕಾರ್", "reference_span": "car", "type": "script"},
            {"hypothesis_span": "hmm", "reference_span": "", "type": "filler"},
            {"hypothesis_span": "vaanginen", "reference_span": "naan", "type": "spelling"},
        ], 60)
        self.assertEqual(out["word_edits"], 0)
        self.assertEqual(len(out["accepted"]), 2)
        self.assertEqual(len(out["rejected"]), 1)

    def test_every_occurrence_is_rewritten(self):
        ref, hyp = "sir ok sir", normalize("சார் ok சார்")
        out = apply_equivalences(ref, hyp, [{"hypothesis_span": "சார்", "reference_span": "sir", "type": "script"}], 60)
        self.assertEqual(out["word_edits"], 0)

    def test_rewrite_without_gain_is_rejected(self):
        # "car" is already correct at its position; rewriting the extra token gains nothing
        out = apply_equivalences("car", normalize("car ಕಾರ್"), [
            {"hypothesis_span": "ಕಾರ್", "reference_span": "car", "type": "script"}], 60)
        self.assertEqual(out["accepted"], [])
        self.assertEqual(out["word_edits"], 1)

    def test_span_must_exist(self):
        out = apply_equivalences("a b c", "a x c", [
            {"hypothesis_span": "zzz", "reference_span": "b", "type": "spelling"}], 60)
        self.assertEqual(out["word_edits"], 1)
        self.assertEqual(len(out["rejected"]), 1)
        self.assertEqual(out["accepted"], [])


if __name__ == "__main__":
    unittest.main()


# --- prediction loading: retries, duplicates, missing rows ------------------
class TestLoadPredictions(unittest.TestCase):
    """A runner appends a retry rather than rewriting a failed attempt, so an id
    can appear twice. Scoring both would weight that utterance twice."""

    def _write(self, text):
        import tempfile
        path = Path(tempfile.mkdtemp()) / "p.csv"
        path.write_text(text)
        return path

    def test_successful_retry_wins_over_earlier_failure(self):
        path = self._write("utterance_id,prediction,status\n"
                           "u1,,error\nu1,the retry worked,ok\n")
        self.assertEqual(load_predictions(path)["u1"], "the retry worked")

    def test_duplicate_without_status_is_an_error(self):
        path = self._write("utterance_id,prediction\nu1,a\nu1,b\n")
        with self.assertRaises(ValueError):
            load_predictions(path)

    def test_missing_utterance_is_refused_by_default(self):
        release = pd.DataFrame({"utterance_id": ["u1", "u2"], "call_id": ["c", "c"],
                                "transcription": ["a", "b"], "keywords": ["", ""]})
        preds = pd.Series({"u1": "a"})
        with self.assertRaises(ValueError):
            align(release, preds)
        # opting in scores the missing row as an empty hypothesis, never drops it
        self.assertEqual(len(align(release, preds, require_complete=False)), 2)


class TestBootstrapEdges(unittest.TestCase):
    """Intervals over slices that have nothing to resample."""

    def _frame(self, keywords=""):
        return pd.DataFrame({
            "utterance_id": ["u1", "u2"], "call_id": ["c1", "c2"],
            "transcription": ["hello world", "hello world"],
            "keywords": [keywords, keywords],
            "prediction": ["hello world", "hello there"]})

    def test_kwer_interval_without_keywords_is_nan_and_quiet(self):
        # NaN is the honest answer; it must not arrive with a numpy warning
        # attached, because a warning in a library is the caller's problem.
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            lo, hi = bootstrap_ci(self._frame(), "kwer", n_boot=20)
        self.assertTrue(math.isnan(lo) and math.isnan(hi))

    def test_single_call_interval_is_degenerate_not_an_error(self):
        df = self._frame()
        df["call_id"] = "c1"
        lo, hi = bootstrap_ci(df, "wer", n_boot=20)
        self.assertEqual(lo, hi)

    def test_same_seed_gives_the_same_interval(self):
        self.assertEqual(bootstrap_ci(self._frame(), "wer", n_boot=50),
                         bootstrap_ci(self._frame(), "wer", n_boot=50))
