"""Per-provider language code mapping and routing.

Every map here is keyed on the ISO-639-1 code the release carries in its
`language` column ("hi", "bn", ...), never on a full language name. A map keyed
on names fails in the worst possible way: looking up "hi" returns nothing, the
row falls back to English, and the run completes with no error anywhere and a
quietly wrong result for every Indic utterance.
"""


# Sarvam Saaras: BCP-47 xx-IN. "unknown" = auto-detect / code-switch.
SAARAS_AUTO_CODE = "unknown"
LANG_TO_SAARAS = {
    "hi": "hi-IN", "bn": "bn-IN", "ta": "ta-IN", "te": "te-IN",
    "kn": "kn-IN", "ml": "ml-IN", "mr": "mr-IN", "gu": "gu-IN",
    "pa": "pa-IN", "od": "od-IN", "en": "en-IN",
}

# Deepgram nova-3 routing.
#
# `language=multi` is a fixed flag, not a language pair: nova-3 multi covers 10
# languages of which Hindi is the only Indic one. So a Bengali/Tamil/Telugu row
# sent to `multi` is not "code-switch aware", it is being decoded by a model
# that does not know the language at all. Route those to their mono tag instead
# and accept that embedded English will be transliterated.
NOVA3_MONO_CODES = frozenset({"hi", "en", "bn", "ta", "te", "kn", "mr", "gu"})
NOVA3_MULTI_CODES = frozenset({"hi", "en"})
DEEPGRAM_MULTI = "multi"


def resolve_deepgram_language(base_language: str, is_code_mix: bool) -> str:
    """Pick the nova-3 `language` value for one row.

    Malayalam is deliberately not handled: `eligibility.py` drops `ml` rows from
    Deepgram entirely, because the model has no Malayalam. Routing them to a
    multilingual mode instead would produce a number that looks like Malayalam
    recognition and is not.
    """
    code = (base_language or "").strip().lower()

    if is_code_mix:
        # Only Hindi/English code-switch is something multi actually models.
        if code in NOVA3_MULTI_CODES:
            return DEEPGRAM_MULTI
        if code in NOVA3_MONO_CODES:
            return code
        return DEEPGRAM_MULTI

    if code in NOVA3_MONO_CODES:
        return code
    return DEEPGRAM_MULTI


def resolve_saaras_language(base_language: str, is_code_mix: bool) -> str:
    """Every row gets its base-language xx-IN tag; unknown codes fall to auto.

    Code-mixed rows previously went to auto-detect. They no longer need to:
    `mode=codemix` is what handles the mixing (English stays Latin, Indic goes
    to native script), and the language tag tells the model which Indic script
    that is. Auto-detect scored the same on script here, so this is the
    consistent choice rather than a corrective one.
    """
    return LANG_TO_SAARAS.get((base_language or "").strip().lower(), SAARAS_AUTO_CODE)


def resolve_elevenlabs_language(base_language: str, is_code_mix: bool) -> str | None:
    """None means: omit `language_code` and let Scribe auto-detect.

    Code-mixed rows used to go to auto-detect. Measured on this corpus, Scribe's
    auto-detect and its tagged mode both produce the correct script 100% of the
    time, so the tag costs nothing and removes a source of variance - and it
    keeps every provider on the same footing: each is told the base language
    wherever its API accepts one.

    Some SDK versions reject `language_code=None`, so callers must omit the
    kwarg rather than pass None through.
    """
    code = (base_language or "").strip().lower()
    # "other" is the corpus's bucket for rows whose language tag did not resolve
    # to one of the nine. It is not an ISO code, so passing it through would be
    # a hard API error on all 87 rows; auto-detect is the honest handling.
    if code in ("", "other", "unknown"):
        return None
    return code


# Human-readable names. Used wherever a *prompt* names the language: an LLM
# reads "Hindi" unambiguously, while "hi" is a coin flip against English "hi"
# and "or"/"od" against the English word "or".
LANGUAGE_NAMES = {
    "hi": "Hindi", "bn": "Bengali", "ta": "Tamil", "te": "Telugu",
    "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "od": "Odia", "or": "Odia", "as": "Assamese",
    "en": "English",
}


def language_name(base_language: str) -> str:
    """ISO-639-1 -> English language name, or "" when unknown."""
    return LANGUAGE_NAMES.get((base_language or "").strip().lower(), "")
