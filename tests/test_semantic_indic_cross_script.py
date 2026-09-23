"""Semantic-WER rules for Indic spans, calibrated on the smoke test.

Two different Indic scripts: never forgiven. Writing one Indian language in
another's script is a script or language-identification error, and the sounds
can match by accident across languages.

One Indic script: consonants must match exactly (vowels may differ, which is
where colloquial spelling lives), and in Indo-Aryan scripts the word ending must
match too, because there it is grammar.
"""
import pytest

from indictelephony_bench.metrics.semantic import _to_ascii, verify_pair


def accepted(hyp, ref, kind="spelling"):
    return verify_pair(hyp, ref, kind, min_similarity=60)[0]


# ---- two different Indic scripts ----------------------------------------
@pytest.mark.parametrize("hyp,ref", [
    ("இந்த", "ಇಂದ"),                       # this (ta) / from (kn)
    ("आम्ही", "আমি"),                       # we (mr) / I (bn)
    ("देखी", "দেখি"),                       # same sounds, wrong script: still an error
    ("ગાડી", "गाड़ी"),
])
@pytest.mark.parametrize("kind", ["spelling", "script"])
def test_two_indic_scripts_never_forgiven(hyp, ref, kind):
    assert not accepted(hyp, ref, kind)


# ---- one Indic script: colloquial spelling is forgiven ---------------------
@pytest.mark.parametrize("hyp,ref", [
    ("எப்படி", "எப்டி"),          # vowel elided
    ("ಸುಮ್ಮನೆ", "ಸುಮ್ನೆ"),
    ("ನಾವು", "ನಾವ್"),             # Dravidian final vowel
    ("இல்லை", "இல்ல"),
    ("ছিল", "ছিলো"),              # Bengali final o, same sound
    ("করব", "করবো"),
    ("ठिक", "ठीक"),               # vowel length
    ("हूं", "हूँ"),                 # anusvara / chandrabindu
])
def test_same_script_colloquial_spelling_forgiven(hyp, ref):
    assert accepted(hyp, ref)


# ---- one Indic script: a different word or form is not ---------------------
@pytest.mark.parametrize("hyp,ref", [
    ("ಮಾಡ್ಸಿದ್ದಾರೆ", "ಮಾಡ್ತಿದ್ದಾರೆ"),    # made to do / is doing: one consonant
    ("बंद", "बंध"),                   # aspiration is kept
    ("तुमचा", "तुमचं"),                # gender agreement
    ("वाड्याचे", "वाड्याचं"),
    ("আজকেই", "আজকে"),                 # today itself / today
    ("बोलतीये", "बोलतेय"),
])
def test_same_script_different_word_or_form_rejected(hyp, ref):
    assert not accepted(hyp, ref)


# ---- split / join ---------------------------------------------------------
@pytest.mark.parametrize("hyp,ref", [
    ("ఎంత ఉంది", "ఎంతుంది"),               # sandhi changes the joining vowel
    ("എവിടെയാണ്", "എവിടെ ആണ്"),           # sandhi inserts the glide y
    ("कौन से", "कौनसे"),
    ("health care", "healthcare"),        # identical letters, joined
])
def test_split_join_forgiven(hyp, ref):
    assert accepted(hyp, ref, "split_join")


@pytest.mark.parametrize("hyp,ref", [
    ("boy cut", "boycott"),               # English: only an identical join counts
    ("credit b", "kreditbee"),
    ("என்ன இது", "ಏನಿದು"),                # two Indic scripts
])
def test_split_join_rejected(hyp, ref):
    assert not accepted(hyp, ref, "split_join")


# ---- English written in an Indic script is unaffected -----------------------
@pytest.mark.parametrize("hyp,ref", [("कार", "car"), ("सर", "sir"), ("லோன்", "loan"), ("டோக்கன்", "token")])
def test_english_in_indic_script_forgiven(hyp, ref):
    assert accepted(hyp, ref, "script")


@pytest.mark.parametrize("word,expected", [("லோன்", "lon"), ("நான்", "nan"), ("அவன்", "avan")])
def test_tamil_alveolar_n_is_not_dropped(word, expected):
    assert _to_ascii(word) == expected
