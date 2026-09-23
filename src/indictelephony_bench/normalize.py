"""The scoring normalizer for WER and CER.

Applied IDENTICALLY to the human annotation and to every provider's output, so
no system is scored against a target the others do not get. It removes
differences that are not transcription errors, and nothing else:

  1. NFKC + lowercase          - "OK" and "ok" are one word
  2. apostrophes deleted       - "It's" -> "its", one token. The annotations carry
                                 no punctuation, so turning it into a space would
                                 make one word into two and cost two errors
  3. other punctuation -> space
  4. numbers canonicalised     - cardinals composed then split to digits, so
                                 "twenty four thousand" == "24,000" == "2 4 0 0 0";
                                 dictated runs ("nine two nine three") stay a
                                 sequence rather than being summed
  5. equivalence map           - a short, explicit list of spellings of the SAME
                                 word (§EQUIV). Not synonyms, not morphology
  6. filler collapse           - "hmmm"/"hm" -> "hmm", "uh"/"ah" -> "ah"

Deliberately NOT done: transliteration between scripts. Folding Indic to Latin
merges genuinely different graphemes (Devanagari अ/आ and चे/छे collapse to one
token) and so forgives real recognition errors. Script convention belongs in the
semantic pass, not in WER.
"""
from __future__ import annotations

import re
import unicodedata

# --- 5. equivalence -------------------------------------------------------
# Every entry is one word with more than one accepted spelling, or a filler with
# no fixed spelling. Counts are occurrences observed as substitutions between
# reference and hypothesis across the corpus, i.e. how often the pair actually
# cost someone an error. Nothing here changes what was said.
EQUIV = {
    # "ok" / "okay" - one word, /oʊˈkeɪ/ either way. 273 observed substitutions
    "okay": "ok", "okey": "ok", "okie": "ok", "oke": "ok", "okk": "ok", "okok": "ok",
    # "mam" / "maam" - one syllable, /mæm/. 242 observed substitutions.
    # "madam" is NOT merged in: it is two syllables, /ˈmædəm/, a different word.
    # The data agrees - madam never swaps with mam, only with its own transliterations.
    "maam": "mam",
    # written abbreviation of a word that is spoken in full
    "rs": "rupees", "oclock": "o clock",
    # same pronunciation, different regional spelling
    "colour": "color",
    # non-lexical fillers with no fixed spelling. Only forms of the SAME sound are
    # grouped: the nasal hum, and the open vowel. "uh" and "ah" are kept apart from
    # "hmm" because they are not the same sound.
    "hm": "hmm", "mm": "hmm", "mmm": "hmm",
    "aa": "ah",
    "haa": "han", "haan": "han",
}

# Explicitly NOT merged, though they show up as frequent substitutions:
#   i/im, you/youre, that/thats  - a contraction carries a real extra word
#   product/products, month/months - morphology is a real difference
#   a brand name and its parts     - a brand split in two is a real error
#   dialed/diled                   - "diled" is a typo in the reference, not a variant
#   mam/madam                      - one syllable vs two: different words
#   uh/ah, huh/ah                  - different vowels, not the same filler
#   inquiry/enquiry                - the first vowel genuinely differs

_UNITS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
          "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
          "seventeen": 17, "eighteen": 18, "nineteen": 19}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lakhs": 100000,
           "crore": 10000000, "crores": 10000000, "million": 1000000,
           "billion": 1000000000}
_FILLER = {"and"}
# Devanagari number words, for Hindi and Marathi. 4.09% of references contain
# one, and a provider that writes the digits instead was being charged an error
# for it - up to 490 rows for Whisper alone. The same composition rules apply, so
# "तीन लाख आठ हजार" and "308000" both reduce to 3 0 8 0 0 0.
#
# Compound 21-99 (इक्कीस, बाईस, …) are single irregular words in Hindi and are
# NOT listed: they are rare in this corpus and each would need its own entry.
# They simply stay as text, which costs nothing beyond the status quo.
_UNITS.update({
    "शून्य": 0, "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पांच": 5, "पाँच": 5,
    "छह": 6, "छै": 6, "छः": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "तेरह": 13, "चौदह": 14, "पंद्रह": 15, "पन्द्रह": 15,
    "सोलह": 16, "सत्रह": 17, "अठारह": 18, "उन्नीस": 19,
    # Marathi
    "दोन": 2, "पाच": 5, "सहा": 6, "नऊ": 9, "दहा": 10, "अकरा": 11, "बारा": 12,
    "तेरा": 13, "चौदा": 14, "पंधरा": 15, "सोळा": 16, "सतरा": 17, "अठरा": 18, "एकोणीस": 19,
})
_TENS.update({
    "बीस": 20, "तीस": 30, "चालीस": 40, "पचास": 50, "पच्चास": 50, "साठ": 60, "सत्तर": 70,
    "अस्सी": 80, "नब्बे": 90,
    # Marathi
    "वीस": 20, "चाळीस": 40, "पन्नास": 50, "ऐंशी": 80, "नव्वद": 90,
})
_SCALES.update({
    "सौ": 100, "हजार": 1000, "हज़ार": 1000, "लाख": 100000, "करोड़": 10000000,
    "करोड": 10000000,
    # Marathi
    "शंभर": 100, "कोटी": 10000000,
})


_NUMWORD = set(_UNITS) | set(_TENS) | set(_SCALES)
# --- ordinals -------------------------------------------------------------
# The annotators write "twenty seventh" and never "27th" (0 occurrences); the
# providers write "27th" 380 times and ordinal words 718 times. Left alone that
# is a systematic penalty for a date said the same way. Ordinals are rewritten
# to their cardinal word so the existing composition handles them:
# "twenty seventh" -> "twenty seven" -> 27, and "27th" -> "27" -> 2 7.
_ORDINAL = {
    "first": "one", "third": "three", "fourth": "four", "fifth": "five",
    "sixth": "six", "seventh": "seven", "eighth": "eight", "ninth": "nine",
    "tenth": "ten", "eleventh": "eleven", "twelfth": "twelve",
    "thirteenth": "thirteen", "fourteenth": "fourteen", "fifteenth": "fifteen",
    "sixteenth": "sixteen", "seventeenth": "seventeen", "eighteenth": "eighteen",
    "nineteenth": "nineteen", "twentieth": "twenty", "thirtieth": "thirty",
    "fortieth": "forty", "fiftieth": "fifty", "sixtieth": "sixty",
    "seventieth": "seventy", "eightieth": "eighty", "ninetieth": "ninety",
    "hundredth": "hundred", "thousandth": "thousand",
}
# "second" is deliberately absent: it is a time unit as often as an ordinal
# ("wait one second"). It is converted only where the context makes it ordinal -
# see _strip_ordinals.
_COUNT_BEFORE_SECOND = set(_UNITS) | {"a", "per", "few", "couple", "some"}
_ORD_NUMERAL = re.compile(r"^(\d+)(st|nd|rd|th)$")


def _strip_ordinals(text: str) -> str:
    toks = text.split()
    out = []
    for i, t in enumerate(toks):
        m = _ORD_NUMERAL.match(t)
        if m:
            out.append(m.group(1)); continue
        if t in _ORDINAL:
            out.append(_ORDINAL[t]); continue
        if t == "second":
            prev = toks[i - 1] if i else ""
            # "one second", "a second", "30 second" -> a duration, leave it alone
            out.append("second" if (prev in _COUNT_BEFORE_SECOND or prev.isdigit()) else "two")
            continue
        out.append(t)
    return " ".join(out)


# Characters folded before tokenising, identically on both sides:
#   apostrophes deleted   - "It's" -> "its", one token (see module docstring)
#   ZWNJ / ZWJ deleted    - they sit INSIDE an Indic word and only steer ligature
#                           shaping; the same word typed with and without one
#                           would otherwise compare as two different tokens
#   Indic digits -> ASCII - "५०" and "50" are the same number, but NFKC does not
#                           fold native digits, so they never matched
_APOS = {ord(c): None for c in "'’ʼ´`‌‍"}
for _zero in (0x0966, 0x09E6, 0x0A66, 0x0AE6, 0x0B66, 0x0BE6, 0x0C66, 0x0CE6, 0x0D66):
    for _d in range(10):
        _APOS[_zero + _d] = ord("0") + _d

# Bump whenever a change here can alter any score. Written into every report so
# numbers produced under different normalizers are never mixed in one table.
NORMALIZER_VERSION = "bench_norm_v2"


def _fold_char(c: str) -> str:
    """Punctuation and separators become spaces. Currency symbols become their own
    token (the rupee sign reads as "rupees", matching the "rs" equivalence), so
    "₹287" scores as "rupees 2 8 7" instead of one unmatchable token."""
    if c == "%":
        return " percent "          # "90%" and "ninety percent" are one thing said
    category = unicodedata.category(c)
    if category[0] in {"P", "Z"}:
        return " "
    if category == "Sc":
        return " rupees " if c == "₹" else (" dollars " if c == "$" else f" {c} ")
    return c


def _is_int(tok: str) -> bool:
    return tok.isdigit()


def _valid_next(prev: str | None, nxt: str) -> bool:
    """Can `nxt` continue the cardinal that `prev` ends?

    English cardinals have a shape. "twenty four thousand" is one number because
    a tens-word may take a unit, and either may take a scale. "five ten five
    twenty" is FOUR numbers - a unit cannot be followed by another unit or a
    tens-word. Composing it greedily gave 5+10+5+20 = 40, which is not what was
    said; the correct reading is 5, 10, 5, 20.
    """
    if prev is None:
        return True
    if nxt in _SCALES:
        return True                      # any number may be scaled
    if _is_int(nxt):
        # A written number may be scaled ("2 lakh 15 thousand"), but two adjacent
        # literals are a dictated sequence ("9 2 9 3"), never one number.
        return prev in _SCALES
    if _is_int(prev):
        return False                     # only a scale (handled above) continues a literal
    if prev in _SCALES:
        return True                      # a scale ends a group; a new one may start
    if prev in _TENS and nxt in _UNITS and _UNITS[nxt] <= 9:
        return True                      # twenty | four
    return False                         # unit -> unit, unit -> tens, teen -> anything else


def _compose(words: list[str]) -> int | None:
    total = current = 0
    seen = False
    for w in words:
        if w in _FILLER:
            continue
        if _is_int(w):
            current += int(w); seen = True
        elif w in _UNITS:
            current += _UNITS[w]; seen = True
        elif w in _TENS:
            current += _TENS[w]; seen = True
        elif w in _SCALES:
            sc = _SCALES[w]
            if sc == 100:
                current = (current or 1) * 100
            else:
                total += (current or 1) * sc
                current = 0
            seen = True
        else:
            return None
    return (total + current) if seen else None


def _drop_decimal_point(text: str) -> str:
    """"one point two five" -> "one two five", so it matches "1.25".

    The dot in "1.25" is punctuation and is already a space by this stage, so the
    written form is a digit sequence; dropping the spoken joiner makes the two
    forms identical. "point" is removed only between two numbers, never in "the
    point is" or "point of contact".
    """
    toks = text.split()
    out = []
    for i, t in enumerate(toks):
        if t == "point" and 0 < i < len(toks) - 1:
            prev, nxt = toks[i - 1], toks[i + 1]
            if (prev in _NUMWORD or prev.isdigit()) and (nxt in _NUMWORD or nxt.isdigit()):
                continue
        out.append(t)
    return " ".join(out)


def canon_numbers(text: str, to_digits: bool = True) -> str:
    """Put spoken and written numbers into one form.

    `to_digits=True` (scoring) emits one digit per token, so a single wrong digit
    costs one token whichever way the number was written. `to_digits=False`
    emits the composed integer, which is what the readable reference column uses.
    """
    toks = text.split()
    out: list[str] = []
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.isdigit() and not (i + 1 < len(toks) and toks[i + 1] in _SCALES):
            out.extend(t if to_digits else [t]); i += 1; continue
        if t in _NUMWORD or t.isdigit():
            j, run, prev = i, [], None
            while j < len(toks):
                w = toks[j]
                if w in _FILLER and run:
                    run.append(w); j += 1; continue
                if (w not in _NUMWORD and not _is_int(w)) or not _valid_next(prev, w):
                    break
                run.append(w); prev = w; j += 1
            while run and run[-1] in _FILLER:
                run.pop(); j -= 1
            digits = [w for w in run if w in _UNITS and _UNITS[w] <= 9 or (_is_int(w) and len(w) == 1)]
            if len(run) >= 3 and len(digits) == len(run):
                s = "".join(str(_UNITS[w]) for w in run)      # dictated string
            else:
                v = _compose(run)
                s = str(v) if v is not None else " ".join(run)
            out.extend(s if (to_digits and s.isdigit()) else [s])
            i = j; continue
        out.append(t); i += 1
    return " ".join(out)


def _collapse_filler(tok: str) -> str:
    """"hmmmm" -> "hmm", "ahhh" -> "ah". Only for pure filler shapes."""
    if re.fullmatch(r"h*m{2,}|m{2,}", tok):
        return "hmm"
    if re.fullmatch(r"a+h+|h+a+", tok):
        return "ah" if tok.startswith("a") else "ha"
    return tok


def normalize(text: str, numbers: bool = True, equiv: bool = True) -> str:
    """Full scoring normalization. Use the same flags on both sides."""
    t = unicodedata.normalize("NFKC", str(text)).translate(_APOS).lower()
    t = "".join(_fold_char(c) for c in t)
    t = re.sub(r"\s+", " ", t).strip()
    if not t:
        return ""
    if numbers:
        t = canon_numbers(_drop_decimal_point(_strip_ordinals(t)), to_digits=True)
    if equiv:
        t = " ".join(EQUIV.get(_collapse_filler(w), _collapse_filler(w)) for w in t.split())
    return t


def reference_numeric(text: str) -> str:
    """The annotation, normalized, with number WORDS written as numerals.

    Readable form of the reference for providers that emit digits - "its twenty
    four thousand" -> "its 24000". Scoring does not need this (both sides go to
    digit sequences), it exists so the numeric variant can be inspected.
    """
    t = unicodedata.normalize("NFKC", str(text)).translate(_APOS).lower()
    t = "".join(_fold_char(c) for c in t)
    t = re.sub(r"\s+", " ", t).strip()
    return canon_numbers(_drop_decimal_point(_strip_ordinals(t)), to_digits=False)
