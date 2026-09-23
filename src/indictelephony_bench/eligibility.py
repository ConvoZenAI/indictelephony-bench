"""Per-provider row eligibility rules for benchmark inference.

Rules are matched on the *set* of language codes in the tag, not the literal
string. The old corpus wrote Indic-first tags (`ml-en`, `mr-hi`); the reviewed
corpus writes English-first in a fixed order (`en-ml`, `hi-mr`). Comparing
literals would silently stop excluding the rows these rules exist to exclude.
"""

# Deepgram has no Malayalam support; skip any row containing it.
DEEPGRAM_EXCLUDED_CODES = frozenset({"ml"})

# Gemini Live degrades badly on Marathi-Hindi mixes specifically.
GEMINI_EXCLUDED_CODE_SETS = (frozenset({"mr", "hi"}),)



def _codes(language_tag: str) -> frozenset[str]:
    return frozenset(
        part for part in (language_tag or "").strip().lower().replace("_", "-").split("-") if part
    )


def is_row_eligible(provider: str, base_language: str, language_tag: str) -> bool:
    """Should this provider be scored on this utterance?

    False only when the row names a language the provider has no model for.
    Scoring it anyway would measure the gap in a vendor's language list rather
    than its recognition, and both belong in a write-up, separately.
    """
    key = provider.lower().strip()
    base = (base_language or "").strip().lower()
    codes = _codes(language_tag)
    if base:
        codes = codes | {base}

    if key == "deepgram":
        if codes & DEEPGRAM_EXCLUDED_CODES:
            return False

    if key == "gemini_realtime":
        if any(codes == excluded for excluded in GEMINI_EXCLUDED_CODE_SETS):
            return False

    return True
