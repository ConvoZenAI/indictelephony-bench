# The metrics

Five metrics, all lower-is-better, all corpus-level. This page says what each
one counts, what it deliberately does not, and where a reimplementation usually
diverges.

## Two rules that apply to all of them

**Rates are corpus-level.** Every rate is `total edits ÷ total reference units`
over the rows being scored — never the mean of per-utterance rates. The two
differ more than people expect here: 30% of this corpus is shorter than two
seconds, so averaging rows would let backchannels outvote sentences. The
per-row rate that `compute_*_counts` returns is for inspecting a single
utterance, not for averaging.

**Utterances are never dropped to improve a score.** A turn the system returned
nothing for is scored as a full deletion. The only rows that leave a system's
denominator are languages it has no model for, through the eligibility rule
below, and that exclusion is reported alongside the score.

## The normalizer

Reference and hypothesis pass through `normalize()` first — the *same* function
on both sides, so a difference of writing convention is not charged as a
recognition error. It does six things:

1. NFKC normalization and lower-casing
2. apostrophes deleted (`it's` → `its`, one token — turning it into a space
   would split one word into two and cost two errors, since the references
   carry no punctuation)
3. other punctuation to space
4. number canonicalization: cardinals are composed then written as digit
   sequences, so `twenty four thousand` = `24,000` = `2 4 0 0 0`; a dictated run
   (`nine two nine three`) stays a sequence rather than being summed
5. a short, explicit list of alternative spellings of the *same* word
   (`okay`→`ok`, `maam`→`mam`) — not synonyms, not morphology
6. filler collapse (`hmmm`/`hm` → `hmm`)

It deliberately **does not transliterate between scripts.** Folding Indic to
Latin merges genuinely different graphemes and so forgives real recognition
errors. Script convention is handled in the semantic pass instead, where each
case is verified individually.

## WER — word error rate

Substitutions + deletions + insertions needed to turn the hypothesis into the
reference, divided by the number of reference words.

## CER — character error rate

The same edits counted over characters, divided by reference characters. It
does not depend on where word boundaries are drawn, which matters for Malayalam,
Kannada and Telugu, where whitespace is a poor boundary and one agglutinated
token counts as a whole word error under WER.

The **CER/WER ratio is informative**: a low ratio means a wrong word is only
slightly wrong — typically an inflected ending — while a high one means the
words are wholly different.

## MER — mixed error rate

Errors over *mixed* reference units. A token containing Latin letters or digits
counts as one unit; a native-script token is split into aksharas, each a base
character with its combining marks and any consonant joined through a virama.

This exists because WER and CER both distort code-mixed text: WER lets a long
Devanagari word cost the same as `ok`, and CER lets the script with more
characters per word dominate the total. For English-only text, MER equals WER.

## KWER — keyword miss rate

The share of curator-tagged reference keywords the hypothesis fails to recover.

- A keyword counts as recovered when its words appear **anywhere** in the
  hypothesis, in order, after the same normalizer. Position does not matter: a
  system can mangle the turn around a keyword and still get credit for it.
- The **last word may be inflected**: an Indic case suffix of up to six
  characters (`website` → `websiteல`, `અમદાવાદ` → `અમદાવાદથી`) or an English
  plural. `car` does not match `card` or `scar`.
- Only keywords that occur **verbatim in the reference** are scored. About 1% of
  curator tags are not in their own transcript — a typo, a paraphrase — and no
  system can produce text the reference does not contain.
- Numbers are matched in both composed and uncomposed form.
- A row with no scorable keyword returns `NaN`, never a free 100%.

**KWER is one minus keyword recall.** There is no insertion term, so a system
that invents keywords is not charged. Read it alongside WER, never instead of it.

## S-WER — semantic WER

WER recomputed after forgiving only *verified* same-meaning variants.

An LLM proposes that two spans are the same word written differently, tagged as
`script`, `spelling`, `split/join`, `filler` or `number`. A deterministic check
then accepts or rejects each proposal:

- never accepts a pair spanning two different Indic scripts
- requires identical consonants within one script
- accepts English written in an Indic script
- accepts a number only when its value is confirmed
- keeps a rewrite only if it **strictly reduces** the edit count

So S-WER can never exceed WER, and a wrong word, wrong number or negation is
never forgiven. Every forgiven edit is recorded per utterance, so a reviewer can
audit the difference rather than trust it.

This is the only metric that calls a model, and the only number that is not
perfectly reproducible: two runs of the judge moved a system by at most 0.48
points. Rows the judge could not answer are scored with nothing forgiven — a
coverage gap can never look like a clean transcript.

## Eligibility

An utterance leaves a system's denominator when any language code in its tag
names a language that system does not support. The rule matches on the *set* of
codes in the tag, not the literal string, because tag order has changed across
corpus versions (`ml-en` vs `en-ml`) and comparing literals would silently stop
excluding the rows the rule exists to exclude.

Use it only for a system that genuinely has no model for a language, and say so
when reporting: a smaller denominator is not a better score. The published
baselines apply it once, for Deepgram Nova-3, which has no Malayalam — 23,067 of
25,393 utterances.

## Where reimplementations usually diverge

If your scorer disagrees with this one, it is almost always one of these:

1. **A mean of per-row rates** instead of a corpus rate.
2. **A different normalizer** — most often one that spells digits out, or one
   that transliterates scripts.
3. **Dropped utterances** — skipping empty hypotheses instead of scoring them as
   deletions shrinks the denominator and flatters the score.
4. **KWER read as an edit rate.** It is a recall miss rate; insertions are free.
