"""Semantic WER / CER: lexical error rates after forgiving only verified,
meaning-preserving surface differences.

Plain WER charges a system for writing a correct word in another valid form: an
English loanword in the local script (`car` / `ಕಾರ್`), a spelling variant, a
compound split differently, a hesitation sound. Semantic WER removes exactly
those differences and nothing else:

1. **Propose.** An LLM judge sees the normalized reference and hypothesis and
   lists candidate equivalences `(hypothesis span, reference span, type)`. It
   never produces a score.
2. **Verify.** Code checks every candidate. Both spans must occur as whole tokens
   in their own text, and the pair must pass a deterministic test for its type:
   matching consonant skeletons after transliteration (`script`, `spelling`), identical
   joined form (`split_join`), digits on exactly one side (`number`), a known
   hesitation sound (`filler`). Anything that fails is rejected and recorded.
3. **Rescore.** Accepted spans are rewritten in the hypothesis (every occurrence),
   and a rewrite is kept only if it strictly reduces the word edit count. WER and CER
   are then recomputed with the same counting as the lexical metrics, so
   `semantic_wer <= wer` and `semantic_cer <= cer` on every row by construction.

The reference is never modified, so every system is scored against the identical
denominator. There is no equivalence type for a wrong word, a wrong number or a
dropped negation, so those errors can never be forgiven. Every accepted and
rejected pair is stored per row, so any score can be audited.

`number` pairs are accepted only when the number words provably have the digit
side's value (English, Hindi, Marathi, or English said in a local script). Number
words of other languages cannot be checked and are rejected unless
SEMANTIC_WER_ALLOW_UNVERIFIED_NUMBERS=true; accepted number pairs are counted in
`semantic_number_pairs` so their effect can be ablated.
"""
from __future__ import annotations

import json
import logging
import math
import re
import unicodedata
from pathlib import Path
from typing import Any

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
from jiwer import process_words
from pydantic import BaseModel
from rapidfuzz import fuzz

from indictelephony_bench.config import get_settings
from indictelephony_bench.judge import identity, is_sampled_in, run_cached_batch
from indictelephony_bench.metrics.cer import char_edit_counts
from indictelephony_bench.metrics.wer import word_edit_counts
from indictelephony_bench.normalize import normalize

LOGGER = logging.getLogger(__name__)

# v2: the prompt asks for a whole number as one span. The cache key hashes the
# version and not the prompt text, so the version must change with the prompt.
SEMANTIC_VERSION = "semantic_equiv_v2"
_PROMPT = (Path(__file__).parent / "prompts" / "semantic_equivalence.txt").read_text(encoding="utf-8")

ALLOWED_TYPES = frozenset({"script", "spelling", "split_join", "number", "filler"})

# Phonetic keys of hesitation sounds, after _phonetic_key's folding
# ("hmm" -> "hm", "ம்ம்" -> "m"). Deliberately excludes "haan"/"ha": those mean yes.
# Indic scripts carry an inherent vowel, so "আহ" romanizes as "aha".
FILLER_KEYS = frozenset({"hm", "m", "ah", "a", "aha", "uh", "uha", "um", "er", "eh", "eha", "oh", "oha"})

_SCHEMES = {
    "DEVANAGARI": sanscript.DEVANAGARI, "BENGALI": sanscript.BENGALI,
    "GURMUKHI": sanscript.GURMUKHI, "GUJARATI": sanscript.GUJARATI,
    "ORIYA": sanscript.ORIYA, "TAMIL": sanscript.TAMIL, "TELUGU": sanscript.TELUGU,
    "KANNADA": sanscript.KANNADA, "MALAYALAM": sanscript.MALAYALAM,
}


class EquivalencePair(BaseModel):
    hypothesis_span: str
    reference_span: str
    type: str


class SemanticEquivalenceResponse(BaseModel):
    equivalences: list[EquivalencePair]


# --------------------------------------------------------------------------
# Deterministic verification
# --------------------------------------------------------------------------
def _script(token: str) -> str:
    for char in token:
        if char.isalpha():
            return unicodedata.name(char, "").split(" ")[0]
    return ""


# The transliterator drops Malayalam chillu letters (final consonants) and renders
# റ്റ ("tta") as "RR"; spell them out in the base script first.
_MALAYALAM_FIXES = str.maketrans({"ൾ": "ള്", "ൺ": "ണ്", "ൻ": "ന്", "ർ": "ര്", "ൽ": "ല്", "ൿ": "ക്"})
# The transliterator has no mapping for Tamil ன (alveolar n) and returns it
# untouched, so the ASCII filter below deleted it: லோன் became "lo", not "lon".
# It is one of the most frequent Tamil letters (நான், அவன்), so every pair with
# it lost a consonant. It is said as n; spell it with ந, which romanizes as "n".
_TAMIL_FIXES = str.maketrans({"ன": "ந"})


def _to_ascii(text: str) -> str:
    """Romanize Indic-script tokens (ITRANS), lowercase, keep [a-z0-9 ] only."""
    parts = []
    for token in text.split():
        scheme = _SCHEMES.get(_script(token))
        if scheme == sanscript.MALAYALAM:
            token = token.translate(_MALAYALAM_FIXES).replace("റ്റ", "ട്ട")
        elif scheme == sanscript.TAMIL:
            token = token.translate(_TAMIL_FIXES)
        elif scheme is None:
            # English orthography: soft c, silent final w ("follow", "law")
            token = re.sub(r"c(?=[eiy])", "s", token.lower())
            token = re.sub(r"(?<=[aeiou])w$", "", token)
        if scheme is not None:
            try:
                token = transliterate(token, scheme, sanscript.ITRANS)
            except Exception:  # noqa: BLE001 - unverifiable token stays as-is
                pass
        parts.append(token)
    key = unicodedata.normalize("NFKD", " ".join(parts).lower())
    key = "".join(char for char in key if char.isascii())
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", key)).strip()


def _phonetic_key(text: str) -> str:
    """Script-independent, coarse phonetic spelling used ONLY to verify a pair.

    Never used for scoring: it merges distinct words on purpose (vowel length,
    aspiration), which is acceptable for "are these two spans the same word"
    when the judge has already claimed they are, and not acceptable for WER.
    """
    key = re.sub(r"[0-9]", "", _to_ascii(text))
    key = key.replace("ph", "f")
    key = re.sub(r"c(?!h)", "k", key)
    key = key.replace("q", "k").replace("x", "ks").replace("w", "v").replace("z", "j")
    key = re.sub(r"(?<=[bcdfgjklmnpqrstvxy])h", "", key)   # aspiration
    key = re.sub(r"([a-z])\1+", r"\1", key)                 # doubled letters
    return re.sub(r"\s+", " ", key).strip()


# Sound classes for CROSS-script pairs only: English spelling is not phonetic
# ("sir" is said "sar"), so compare consonant skeletons with voicing and
# sibilants merged. After _phonetic_key, a remaining "c" is always "ch".
# Tamil ச romanizes as "ch"/"jh" but is said s/ch/j; Bengali ব romanizes as "v" but
# is said b; so j joins the sibilants and v joins the labials.
_SOUND_CLASSES = str.maketrans({"c": "s", "j": "s", "g": "k", "d": "t", "b": "p", "f": "p", "v": "p", "y": None})


def _skeleton(text: str, sound_classes: bool) -> str:
    """Consonants only, nasals merged, repeats collapsed.

    Spelling variants of one word differ in vowels ("யத"/"যতো", "அஞ்சு"/"அஞ்சி")
    and in nasal assimilation ("எண்பத்தி"/"எம்பத்தி"); different words almost
    always differ in consonants ("કહી"/"કઈ", "શું"/"સુ").
    """
    key = _phonetic_key(text).replace(" ", "")
    if sound_classes:
        key = key.translate(_SOUND_CLASSES)
    key = re.sub(r"[aeiou]", "", key).replace("m", "n")
    return re.sub(r"([a-z])\1+", r"\1", key)


def _similarity(a: str, b: str) -> float:
    ka, kb = _phonetic_key(a).replace(" ", ""), _phonetic_key(b).replace(" ", "")
    if not ka or not kb:
        return 0.0
    return fuzz.ratio(ka, kb)


def _skeleton_match(a: str, b: str, sound_classes: bool, threshold: float) -> bool:
    sa, sb = _skeleton(a, sound_classes), _skeleton(b, sound_classes)
    if not sa or not sb:
        return False
    if max(len(sa), len(sb)) <= 3:
        return sa == sb          # short words: one consonant is the whole word
    return fuzz.ratio(sa, sb) >= threshold


# Scripts of Indo-Aryan languages, where a word's final vowel is often grammar
# (gender, number, emphasis) rather than a free spelling choice.
_ENDING_IS_GRAMMAR = frozenset({"DEVANAGARI", "BENGALI", "GUJARATI"})


def _consonant_key(text: str) -> str:
    """Consonants of a span in its own script, exactly, joined across spaces.

    Folds only what the script leaves ambiguous: vowels (colloquial speech drops
    and changes them: எப்படி/எப்டி, ಸುಮ್ಮನೆ/ಸುಮ್ನೆ), anusvara against a nasal
    consonant (m/n), the glide y (sandhi inserts it: നാൽപ്പത്തിയൊന്ന്/നാൽപ്പത്തി
    ഒന്ന്), doubled consonants, and ph/f (फ/फ़). Aspiration is KEPT, and nothing is
    fuzzy: one differing consonant is a different word (ಮಾಡ್ಸಿದ್ದಾರೆ, made to do,
    against ಮಾಡ್ತಿದ್ದಾರೆ, is doing).
    """
    key = _to_ascii(text).replace(" ", "").replace("ph", "f").replace("m", "n").replace("y", "")
    key = re.sub(r"[aeiou]", "", key)
    return re.sub(r"([a-z]h?)\1+", r"\1", key)


def _final_vowel(text: str, script: str) -> str:
    tokens = _to_ascii(text).split()
    token = tokens[-1].replace("m", "n") if tokens else ""
    if len(token) > 2:
        token = re.sub(r"a$", "", token)            # the inherent vowel, written or not
    ending = re.search(r"[aeiou]*n?$", token).group()
    if script == "BENGALI":
        # Bengali spells a final "o" or leaves it to the inherent vowel
        # (ছিল/ছিলো, করব/করবো, এখনও/এখনো): the same sound, so it is folded.
        ending = ending.replace("a", "")
        return "" if ending in ("", "o") else ending
    return ending


def _same_script_indic_match(a: str, b: str) -> tuple[bool, str]:
    if _consonant_key(a) != _consonant_key(b):
        return False, "a consonant differs"
    script = _script(a)
    if script in _ENDING_IS_GRAMMAR and _final_vowel(a, script) != _final_vowel(b, script):
        # तुमचा/तुमचं, वाड्याचे/वाड्याचं, আজকেই/আজকে: the ending is grammar
        return False, "word ending differs: a grammatical form, not a spelling"
    return True, "ok"


def _has_digit(text: str) -> bool:
    return any(token.isdigit() for token in text.split())


def _digits(text: str) -> str:
    return "".join(token for token in text.split() if token.isdigit())


# English number words as they come out of romanizing an Indian script: a
# provider that hears "two hundred" in a Kannada call may write ಟೂ ಹಂಡ್ರೆಡ್, which
# romanizes to "tu hamdred", not "two hundred". _english_number_key folds the
# systematic differences; this table maps each folded key to the English word,
# and the scoring normalizer then composes the value.
#
# Built from the spellings observed in the semantic-WER smoke test. It is a
# closed vocabulary on purpose: a span is read as English numbers only if EVERY
# token maps, so a native word can never slip through, and the value must still
# equal the digit side exactly. Forms with a case suffix (Bengali থাউজেন্ডের,
# "of thousand") are deliberately absent.
_ENGLISH_NUMBER_WORDS = {
    "ziro": "zero", "zero": "zero",
    "van": "one", "on": "one", "one": "one",
    "tu": "two", "two": "two",
    "tri": "three", "three": "three",
    "for": "four", "four": "four",
    "faiv": "five", "five": "five",
    "siks": "six", "six": "six",
    "sevan": "seven", "seven": "seven",
    "et": "eight", "eit": "eight", "eyt": "eight", "eight": "eight",
    "nain": "nine", "nine": "nine",
    "ten": "ten",
    "tvelv": "twelve", "twelve": "twelve",
    "itin": "eighteen", "eighteen": "eighteen",
    "tventi": "twenty", "tvanti": "twenty", "tvanri": "twenty", "toyaenti": "twenty",
    "tuyaenti": "twenty", "twenty": "twenty",
    "tarti": "thirty", "thirty": "thirty",
    "farti": "forty", "forti": "forty", "forty": "forty",
    "fifti": "fifty", "fifty": "fifty",
    "siksti": "sixty", "siksati": "sixty", "sixty": "sixty",
    "eiti": "eighty", "eti": "eighty", "eighty": "eighty",
    "nainti": "ninety", "nainati": "ninety", "ninety": "ninety",
    "handred": "hundred", "handrad": "hundred", "hundred": "hundred",
    "tausand": "thousand", "tasand": "thousand", "taujand": "thousand", "taujyand": "thousand",
    "thousand": "thousand",
    "lakh": "lakh", "kror": "crore",
    "point": "point", "paint": "point", "payint": "point",
    "and": "and", "or": "or", "ar": "or",
    "rupis": "rupees", "rupiz": "rupees", "rupees": "rupees",
    "parsent": "percent", "persent": "percent", "percent": "percent",
}


def _english_number_key(token: str) -> str:
    """Fold a romanized token so its English-number spellings coincide.

    ph -> f (ফাইভ "phaibha"), bh/vh -> v, jh -> z and j before "iro" -> z
    (ঝীরো "jhiro"), th -> t (থ্রি "thri"), a nasal before a stop -> n (the
    anusvara in ಹಂಡ್ರೆಡ್ "hamdred"), the romanized inherent vowel "a" dropped
    from the end (ಸೆವನ "sevana"), then doubled letters collapsed (এট্টি "etti").
    """
    t = token.lower()
    t = t.replace("ph", "f").replace("bh", "v").replace("vh", "v").replace("jh", "z")
    t = re.sub(r"^j(?=iro)", "z", t).replace("th", "t")
    t = re.sub(r"m(?=[dtkgpb])", "n", t)
    if len(t) > 2:
        t = re.sub(r"a$", "", t)
    return re.sub(r"([a-z])\1+", r"\1", t)


def _as_english_numbers(word_span: str) -> str | None:
    """The span rewritten as English number words, or None if any token is not one."""
    words = [_ENGLISH_NUMBER_WORDS.get(_english_number_key(t)) for t in _to_ascii(word_span).split()]
    return " ".join(words) if words and all(words) else None


def _number_verified(digit_span: str, word_span: str) -> bool:
    """True when the number words provably have the digit side's value.

    The scoring normalizer composes English, Hindi and Marathi number words, so
    this checks the words as written, romanized, and read as English number words
    written in a local script ("ಟೂ ಹಂಡ್ರೆಡ್" = two hundred). Native number words of
    Bengali, Gujarati, Kannada, Malayalam, Tamil and Telugu are not checked: their
    composition is agglutinative and colloquial, a hand-built reader would be
    error-prone, and an unverified pair is never forgiven.
    """
    target = _digits(digit_span)
    rest = [t for t in digit_span.split() if not t.isdigit()]
    candidates = [word_span, _to_ascii(word_span)]
    english = _as_english_numbers(word_span)
    if english:
        candidates.append(english)
    for candidate in candidates:
        normalized = normalize(candidate)
        if _digits(normalized) == target and [t for t in normalized.split() if not t.isdigit()] == rest:
            return True
    return False


def _is_filler(text: str) -> bool:
    tokens = text.split()
    return bool(tokens) and all(_phonetic_key(token) in FILLER_KEYS for token in tokens)


def _contains(tokens: list[str], span: list[str]) -> int | None:
    n = len(span)
    for start in range(len(tokens) - n + 1):
        if tokens[start : start + n] == span:
            return start
    return None


def verify_pair(
    hyp_span: str,
    ref_span: str,
    kind: str,
    min_similarity: float,
    allow_unverified_numbers: bool = False,
) -> tuple[bool, str]:
    """Deterministic check that a proposed pair really is one word written two ways.

    Spans must already be normalized. Returns (accepted, reason).
    """
    kind = (kind or "").strip().lower()
    if kind not in ALLOWED_TYPES:
        return False, f"type {kind!r} not allowed"
    if hyp_span == ref_span:
        return False, "spans identical"

    if kind == "filler":
        if hyp_span and ref_span:
            return False, "filler must be absent on one side"
        present = hyp_span or ref_span
        return (True, "ok") if _is_filler(present) else (False, "not a hesitation sound")

    if not hyp_span or not ref_span:
        return False, "empty span"

    if kind == "number":
        if _has_digit(hyp_span) == _has_digit(ref_span):
            return False, "number pair needs digits on exactly one side"
        digit_side, word_side = (hyp_span, ref_span) if _has_digit(hyp_span) else (ref_span, hyp_span)
        if len(word_side.split()) > 8:
            return False, "number-word span too long"
        if _number_verified(digit_side, word_side):
            return True, "ok"
        return (True, "unverified number") if allow_unverified_numbers else (False, "number value not verifiable")

    if _has_digit(hyp_span) or _has_digit(ref_span):
        return False, "digits only allowed in number pairs"

    if kind == "split_join":
        if len(hyp_span.split()) == len(ref_span.split()):
            return False, "split_join needs different token counts"
        if hyp_span.replace(" ", "") == ref_span.replace(" ", ""):
            return True, "ok"
        # Only an English-Indic pair may join to something not letter-identical,
        # because transliteration changes the letters. Within one script a join
        # that is not identical is a different word ("boy cut" / "boycott"), and
        # two Indic scripts is a script error, not a split.
        scripts = {_script(t) for t in (hyp_span + " " + ref_span).split()} - {""}
        indic = scripts - {"LATIN"}
        if len(indic) > 1:
            return False, "two different Indic scripts: a script error, not a split"
        if scripts == {"LATIN"}:
            return False, "joined forms differ"     # English: only an identical join counts
        if "LATIN" not in scripts:
            # one Indic script: sandhi changes vowels where words join
            # (ఎంత ఉంది / ఎంతుంది), so compare consonants and ending as for spelling
            return _same_script_indic_match(hyp_span, ref_span)
        joined_a, joined_b = hyp_span.replace(" ", ""), ref_span.replace(" ", "")
        if _skeleton_match(joined_a, joined_b, sound_classes=True, threshold=90) and _similarity(joined_a, joined_b) >= 80:
            return True, "ok"
        return False, "joined forms differ"

    if abs(len(hyp_span.split()) - len(ref_span.split())) > 1:
        return False, "token counts differ too much"
    same_script = _script(hyp_span) == _script(ref_span)

    # An Indian language written in another Indian script is a script or
    # language-identification error, not a spelling of the same word: the reader
    # of a Kannada transcript cannot read Tamil script. The sounds can also match
    # by accident across languages: Tamil இந்த ("indha", this) against Kannada
    # ಇಂದ ("inda", from), Marathi आम्ही ("amhi", we) against Bengali আমি ("ami", I).
    # So these pairs are never forgiven.
    both_indic = "LATIN" not in (_script(hyp_span), _script(ref_span))
    if kind in ("script", "spelling") and not same_script and both_indic:
        return False, "two different Indic scripts: a script error, not a variant"

    if kind == "script":
        if same_script:
            return False, "script pair in the same script"
        if _skeleton_match(hyp_span, ref_span, sound_classes=True, threshold=75):
            return True, "ok"
        score = _similarity(hyp_span, ref_span)
        if score >= min_similarity and _skeleton_match(hyp_span, ref_span, sound_classes=True, threshold=50):
            return True, "ok"
        return False, f"consonants differ (phonetic similarity {score:.0f})"

    # spelling: same consonants, vowels may differ.
    #
    # That rule is right for Indic scripts, where the same word is routinely
    # written with a different vowel diacritic ("नहीं"/"नही"), but wrong for
    # English, where the vowel IS the difference between two words: "loan"/"lane",
    # "bill"/"bull", "month"/"mouth" and "insurance"/"assurance" all share their
    # consonants. No similarity threshold separates those from genuine variants -
    # "insurance"/"assurance" scores 82 while "grey"/"gray" scores 75 - so for two
    # Latin spans we do not guess. The English spelling variants worth forgiving
    # are few and enumerable, and the scoring normalizer has already merged them
    # from its explicit list before this point, so anything still differing here
    # is a different word.
    if same_script and _script(hyp_span) == "LATIN":
        return False, "English spelling variants are handled by the normalizer's list"
    if not same_script:
        return False, "spelling pair across scripts"
    # Same Indic script. The consonants must match exactly (a fuzzy match accepted
    # ಮಾಡ್ಸಿದ್ದಾರೆ for ಮಾಡ್ತಿದ್ದಾರೆ), and in Indo-Aryan scripts the word ending too
    # (a consonant match accepted तुमचा for तुमचं). Vowels elsewhere may differ:
    # that is where colloquial spelling lives.
    return _same_script_indic_match(hyp_span, ref_span)


def apply_equivalences(
    reference: str,
    hypothesis: str,
    pairs: list[dict[str, Any]],
    min_similarity: float,
    allow_unverified_numbers: bool = False,
) -> dict[str, Any]:
    """Apply verified pairs to the hypothesis; never touches the reference.

    `reference` and `hypothesis` must already be normalized.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    current, _ = word_edit_counts(reference, hypothesis)
    accepted: list[dict[str, str]] = []
    rejected: list[dict[str, str]] = []

    for pair in pairs:
        hyp_span = normalize(pair.get("hypothesis_span", "") or "")
        ref_span = normalize(pair.get("reference_span", "") or "")
        kind = str(pair.get("type", "")).strip().lower()
        record = {"hypothesis_span": hyp_span, "reference_span": ref_span, "type": kind}

        ok, reason = verify_pair(hyp_span, ref_span, kind, min_similarity, allow_unverified_numbers)
        if ok and ref_span and _contains(ref_tokens, ref_span.split()) is None:
            ok, reason = False, "reference span not in reference"
        if ok and hyp_span and _contains(hyp_tokens, hyp_span.split()) is None:
            ok, reason = False, "hypothesis span not in hypothesis"
        if not ok:
            rejected.append({**record, "reason": reason})
            continue

        if hyp_span:
            # Rewrite every occurrence whose rewrite strictly reduces the edit
            # count ("sir ... sir" written twice in the other script).
            span, replacement, applied_any = hyp_span.split(), ref_span.split(), False
            progress = True
            while progress:
                progress = False
                for start in range(len(hyp_tokens) - len(span) + 1):
                    if hyp_tokens[start : start + len(span)] != span:
                        continue
                    trial = hyp_tokens[:start] + replacement + hyp_tokens[start + len(span) :]
                    edits, _ = word_edit_counts(reference, " ".join(trial))
                    if edits < current:
                        hyp_tokens, current, applied_any, progress = trial, edits, True, True
                        break
            if applied_any:
                accepted.append(record)
            else:
                rejected.append({**record, "reason": "rewrite does not reduce edits"})
            continue
        # A filler present only in the reference: insert it where the
        # alignment deletes it, so the reference itself stays untouched.
        candidate = None
        if hyp_tokens:
            alignment = process_words(" ".join(ref_tokens), " ".join(hyp_tokens)).alignments[0]
            for chunk in alignment:
                if chunk.type == "delete" and ref_tokens[chunk.ref_start_idx : chunk.ref_end_idx] == ref_span.split():
                    candidate = hyp_tokens[: chunk.hyp_start_idx] + ref_span.split() + hyp_tokens[chunk.hyp_start_idx :]
                    break
        if candidate is None:
            rejected.append({**record, "reason": "no aligned deletion for reference filler"})
            continue

        edits, _ = word_edit_counts(reference, " ".join(candidate))
        if edits >= current:
            rejected.append({**record, "reason": "rewrite does not reduce edits"})
            continue
        hyp_tokens, current = candidate, edits
        accepted.append(record)

    return {"hypothesis": " ".join(hyp_tokens), "word_edits": current,
            "accepted": accepted, "rejected": rejected}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------
def _model() -> str:
    return get_settings().judge_model


def _prompt(reference: str, hypothesis: str, language: str) -> str:
    payload = {"language": language, "reference": reference, "hypothesis": hypothesis}
    return _PROMPT + "\n\nINPUT:\n" + json.dumps(payload, ensure_ascii=False, indent=2)


def _result(status: str, wer_edits: float = math.nan, cer_edits: float = math.nan,
            wer_len: int = 0, cer_len: int = 0, accepted=None, rejected=None) -> dict[str, Any]:
    accepted = accepted or []
    return {
        "semantic_wer": (wer_edits / wer_len) if wer_len and not math.isnan(wer_edits) else math.nan,
        "semantic_cer": (cer_edits / cer_len) if cer_len and not math.isnan(cer_edits) else math.nan,
        "semantic_wer_edits": wer_edits,
        "semantic_cer_edits": cer_edits,
        "semantic_status": status,
        "semantic_number_pairs": sum(1 for p in accepted if p.get("type") == "number"),
        "semantic_accepted": json.dumps(accepted, ensure_ascii=False) if accepted else "",
        "semantic_rejected": json.dumps(rejected, ensure_ascii=False) if rejected else "",
    }


def evaluate_semantic_wer(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One result per input row, same order.

    `rows` need `ground_truth`, `hypothesis`, `language`, `row_key`.
    """
    settings = get_settings()
    try:
        settings.require_judge_key()
        has_credentials = True
        LOGGER.info("Semantic WER judge: %s", _model())
    except RuntimeError as exc:
        has_credentials = False
        # Say so once, loudly. A run with no key still returns a result for every
        # row - the rows a judge never saw come back NaN rather than 0, so a
        # missing key can never be mistaken for "nothing needed forgiving".
        LOGGER.warning("%s Semantic WER/CER will be NaN except for rows settled "
                       "without a judge.", exc)

    results: list[dict[str, Any] | None] = [None] * len(rows)
    prepared: dict[int, tuple[str, str, str]] = {}
    prompts: dict[str, str] = {}

    for index, row in enumerate(rows):
        reference = normalize(row.get("ground_truth") or "")
        hypothesis = normalize(row.get("hypothesis") or "")
        wer_edits, wer_len = word_edit_counts(reference, hypothesis)
        cer_edits, cer_len = char_edit_counts(reference, hypothesis)

        if not reference:
            results[index] = _result("empty_reference")
        elif hypothesis == reference:
            results[index] = _result("exact", 0, 0, wer_len, cer_len)
        elif not hypothesis:
            # Nothing written, nothing to forgive: semantic equals lexical.
            results[index] = _result("empty_hypothesis", wer_edits, cer_edits, wer_len, cer_len)
        elif not is_sampled_in(str(row.get("row_key", index)), settings.semantic_sample_rate):
            results[index] = _result("sampled_out")
        elif not has_credentials:
            results[index] = _result("no_api_key")
        else:
            language = str(row.get("language") or "")
            key = identity((reference, hypothesis, language, SEMANTIC_VERSION))
            prepared[index] = (key, reference, hypothesis)
            prompts.setdefault(key, _prompt(reference, hypothesis, language))

    if prompts:
        responses, _ = run_cached_batch(
            list(prompts.items()),
            SemanticEquivalenceResponse,
            f"semantic_equivalence_{SEMANTIC_VERSION}",
            model=_model(),
        )
        for index, (key, reference, hypothesis) in prepared.items():
            response = responses.get(key)
            if response is None:
                results[index] = _result("judge_error")
                continue
            wer_edits, wer_len = word_edit_counts(reference, hypothesis)
            cer_edits, cer_len = char_edit_counts(reference, hypothesis)
            applied = apply_equivalences(reference, hypothesis, list(response.get("equivalences", [])),
                                         settings.semantic_min_similarity,
                                         settings.semantic_allow_unverified_numbers)
            sem_cer, _ = char_edit_counts(reference, applied["hypothesis"])
            results[index] = _result(
                "success" if applied["accepted"] else "no_equivalences",
                applied["word_edits"], min(sem_cer, cer_edits), wer_len, cer_len,
                applied["accepted"], applied["rejected"],
            )

    return [r if r is not None else _result("unknown") for r in results]
