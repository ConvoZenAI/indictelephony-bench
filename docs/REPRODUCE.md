# Reproducing the published numbers

This page is deliberately specific about what the public artifacts do and do not
let you reproduce. A benchmark that implies one-command replication it cannot
deliver wastes the reader's afternoon.

## What you can reproduce exactly

**Everything about the corpus.** Utterance and hour counts, per-language and
per-domain breakdowns, the duration histogram, code-mixing share, keyword
coverage. All of it is a function of the released dataset alone:

```python
from indictelephony_bench import load_release

df = load_release("all")
print(len(df), df.duration_sec.sum() / 3600, df.call_id.nunique())
# 25393 30.175... 320
print(100 * df.is_code_mixed.mean())          # 70.4
print(100 * (df.duration_sec < 2).mean())     # 30.0
```

**Every metric, on any transcripts you have.** The normalizer and the five
metrics are here in full, so a score you compute is the score we would compute.

**The normalizer, against the release.** The dataset ships
`transcription_normalized`, which is `normalize(transcription)`. That is a
direct check that your installation reproduces ours, with no model and no keys:

```python
from indictelephony_bench import load_release, normalize

df = load_release("Tamil")
assert (df.transcription.map(normalize) == df.transcription_normalized).all()
```

If that assertion holds, your scorer and ours agree on the hard part.

## What you cannot reproduce from the public artifacts alone

**The five vendors' transcripts are not released.** Publishing another company's
model output raises licensing questions under their terms of service that we
have not resolved, so the honest position is that we do not distribute them.

This means the published baseline table — Akshara 9.7, Saaras v4 16.8, Scribe v2
23.6, Pulse 27.4, Nova-3 31.5 — is reproduced by **re-running those systems**,
not by re-scoring stored output. That costs API calls and money, and the vendors'
models move, so a rerun months later measures a different model with the same
name. Treat the published figures as a dated measurement, which is what they are:
the systems were run between August and September 2026, with the model
identifiers and request settings listed in the paper.

Every system is re-run the same way, each through its public interface and a key
you supply:

| System | Install | Key |
|---|---|---|
| Akshara | `pip install -e ".[akshara]"` (the [`convozen`](https://pypi.org/project/convozen/) SDK) | `CONVOZEN_API_KEY` |
| Saaras v4 | `pip install -e ".[saaras]"` | `SARVAM_API_KEY` |
| Scribe v2 | `pip install -e ".[elevenlabs]"` | `ELEVENLABS_API_KEY` |
| Pulse | `pip install -e ".[smallest]"` | `SMALLEST_API_KEY` |
| Nova-3 | `pip install -e ".[deepgram]"` | `DEEPGRAM_API_KEY` |

```bash
python scripts/run_inference.py --provider akshara --audio-dir ./audio
```

Akshara is reached through the public SDK like the others, so its row is held to
the same standard: anyone with a key can rerun it, and the same caveat applies -
the SDK serves the current `akshara-pro` model, which may not be the model the
paper measured.

**Semantic WER is not bit-reproducible**, because an LLM proposes the candidate
pairs. Two runs of the judge over the same data moved a system by at most 0.48
points. The deterministic verifier that accepts or rejects each proposal *is*
reproducible, and every forgiven edit is recorded per utterance.

## Scoring a system you run yourself

```bash
pip install -e ".[hub]"
python scripts/score.py --predictions my_system.csv --by language --ci
```

The comparison that holds is **between systems within a language**. Each
language was recorded over a limited set of calls, so a cross-language
difference also varies speaker and line.

## Reporting a result

If you publish a number from this benchmark, please state:

1. **Which release** — v1.0 (28,111 utterances) or v1.0.1 (25,393). They are
   both frozen and both valid; scores differ by 0.26 to 0.55 WER points.
2. **How many utterances you scored**, and how you treated turns the system
   returned nothing for.
3. **Whether you applied the eligibility rule**, and to which languages.
4. **When you ran the system**, if it is a hosted API.

A number without (1) and (2) cannot be compared to anything.

## Version pinning

```python
load_release("all", revision="v1.0.1")
```

The dataset repository is tagged per release, so a pinned revision keeps a
result checkable after the dataset moves on.
