"""Keyword recall and keyword WER.

Each row's curator keywords should be verbatim spans of its reference. A keyword
counts as recovered when it appears in the hypothesis after the SAME scoring
normalizer used for WER.

Which keywords are scored: only those that actually occur in the reference
(when the reference is passed). About 1% of curator keywords are not in their
own transcript - typos ("white filed" for "Whitefield"), a different inflection,
a paraphrase - and no system can be expected to produce text the reference does
not contain.

Matching rule, per keyword:
  * every word but the last must match a hypothesis token exactly, in order;
  * the last word may match a LONGER hypothesis token when the extra characters
    are an inflection: an Indic case suffix glued to the word (`website` ->
    `websiteல`, `અમદાવાદ` -> `અમદાવાદથી`, at most MAX_INDIC_SUFFIX characters,
    keyword at least MIN_PREFIX_CHARS long), or an English plural (`loan` ->
    `loans`). `car` does not match `card` or `scar`.
  * numbers are tried in both forms. The normalizer composes adjacent number
    words into one value, so "eighty thousand" inside "eighty thousand eight
    lakhs" composes differently in context than alone; matching either the
    composed (digits) or the uncomposed (words) form handles that, and still
    lets "twenty four thousand" match a hypothesis that wrote "24,000".

Corpus-level keyword recall is total hits / total scored keywords; keyword WER is
1 - recall. Rows with no scorable keywords return NaN, never a free perfect score.
"""
import math
import re

from indictelephony_bench.normalize import normalize

# Separators actually used by curators: comma and newline. A comma between two
# digits is a digit-group separator ("24,000"), not a keyword separator.
_SEPARATORS = re.compile(r"\n+|(?<!\d),|,(?!\d)")
MIN_PREFIX_CHARS = 3
MAX_INDIC_SUFFIX = 6
_ENGLISH_INFLECTIONS = frozenset({"s", "es"})


def _forms(text: str) -> tuple[str, ...]:
    """Normalized forms of a text: numbers composed to digits, and left as words."""
    composed = normalize(text)
    words = normalize(text, numbers=False)
    return (composed,) if composed == words else (composed, words)


def parse_keywords(raw_keywords: object) -> list[str]:
    """Split, normalize and de-duplicate a curator keyword cell."""
    if raw_keywords is None:
        return []
    text = str(raw_keywords)
    if text.strip().lower() in {"", "nan", "none"}:
        return []
    keywords: list[str] = []
    for item in _SEPARATORS.split(text):
        keyword = normalize(item)
        if keyword and keyword not in keywords:
            keywords.append(keyword)
    return keywords


def _parse_raw(raw_keywords: object) -> list[str]:
    """Split and de-duplicate, keeping the raw text so both number forms can be built."""
    if raw_keywords is None:
        return []
    text = str(raw_keywords)
    if text.strip().lower() in {"", "nan", "none"}:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for item in _SEPARATORS.split(text):
        key = normalize(item)
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _last_token_matches(keyword_last: str, token: str) -> bool:
    if token == keyword_last:
        return True
    if len(keyword_last) < MIN_PREFIX_CHARS or not token.startswith(keyword_last):
        return False
    rest = token[len(keyword_last):]
    if rest.isascii():
        return rest in _ENGLISH_INFLECTIONS
    return len(rest) <= MAX_INDIC_SUFFIX and not any(ch.isascii() and ch.isalpha() for ch in rest)


def keyword_in(keyword: str, hypothesis_tokens: list[str]) -> bool:
    """Does this keyword appear anywhere in the hypothesis, in order?

    Position in the turn does not matter: a system may mangle the words around a
    keyword and still have produced the keyword.
    """
    words = keyword.split()
    n = len(words)
    if not n:
        return False
    head, last = words[:-1], words[-1]
    for start in range(len(hypothesis_tokens) - n + 1):
        window = hypothesis_tokens[start : start + n]
        if window[:-1] == head and _last_token_matches(last, window[-1]):
            return True
    return False


def _found(raw_keyword: str, text_forms: tuple[list[str], ...]) -> bool:
    return any(keyword_in(form, tokens) for form in _forms(raw_keyword) for tokens in text_forms)


def _token_forms(text: str) -> tuple[list[str], ...]:
    return tuple(form.split() for form in _forms(text or ""))


def compute_keyword_counts(
    raw_keywords: object,
    prediction: str,
    reference: str | None = None,
) -> tuple[int, int]:
    """(keywords recovered, keywords scored). (0, 0) when nothing is scorable.

    With `reference`, keywords that do not occur in the reference are not scored.
    """
    keywords = _parse_raw(raw_keywords)
    if reference is not None:
        reference_forms = _token_forms(reference)
        keywords = [k for k in keywords if _found(k, reference_forms)]
    if not keywords:
        return 0, 0
    hypothesis_forms = _token_forms(prediction)
    return sum(_found(k, hypothesis_forms) for k in keywords), len(keywords)


def compute_keyword_wer(
    raw_keywords: object,
    prediction: str,
    threshold: int | None = None,
    reference: str | None = None,
) -> float:
    """Row-level keyword miss rate, NaN when the row has no scorable keywords.

    `threshold` is accepted for backward compatibility and ignored: matching is
    no longer fuzzy.
    """
    hits, total = compute_keyword_counts(raw_keywords, prediction, reference)
    if not total:
        return math.nan
    return 1.0 - hits / total
