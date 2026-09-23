# IndicTelephony-Bench

[![Dataset on Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20dataset-IndicTelephony--Bench-yellow)](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench)
[![License](https://img.shields.io/badge/code-Apache--2.0-blue)](LICENSE)
[![Data licence](https://img.shields.io/badge/data-CC--BY--4.0-blue)](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![ConvoZen SDK on PyPI](https://img.shields.io/pypi/v/convozen?label=ConvoZen%20SDK&logo=pypi&logoColor=white)](https://pypi.org/project/convozen/)

Scoring for **IndicTelephony-Bench**, an open ASR benchmark of Indian telephone
speech: 25,393 human-curated utterances, 30.18 hours, nine languages, recorded
over live SIP lines at 8 kHz and 70.4% code-mixed.

The dataset is published separately, on the Hugging Face hub. This repository is
the other half: the normalizer, the five metrics and the call-level bootstrap,
so that every number in the paper and the article can be recomputed rather than
taken on trust — and so you can score your own system on exactly the same terms.

```bash
pip install -e ".[hub]"
python scripts/score.py --predictions my_system.csv --by language --ci
```

```
                          WER     CER     MER    KWER     utts
overall                  9.73%   5.59%   8.48%   8.97%   25,393
  95% CI (WER)           9.00% - 10.50%   (1,000 resamples over 320 calls)
```

## What is here

| | |
|---|---|
| `src/indictelephony_bench/normalize.py` | the scoring normalizer, applied identically to reference and hypothesis |
| `src/indictelephony_bench/metrics/` | WER, CER, MER, KWER and semantic WER |
| `src/indictelephony_bench/scoring.py` | corpus-level aggregation and call-level bootstrap intervals |
| `src/indictelephony_bench/data.py` | loading the release and your predictions |
| `src/indictelephony_bench/eligibility.py` | the rule for languages a system does not support |
| `scripts/score.py` | the command-line scorer |
| `docs/METRICS.md` | what each metric counts, and what it deliberately does not |
| `docs/REPRODUCE.md` | reproducing the published tables |

## Scoring your own system

Produce a CSV with one row per released utterance:

```csv
utterance_id,prediction
call_0033_chunk0050,respective father name is pratap nair pradeep nair yes yes yes mam
call_0034_chunk0033,hmm
```

Then:

```bash
python scripts/score.py --predictions my_system.csv            # all nine languages
python scripts/score.py --predictions my_system.csv --config Tamil
python scripts/score.py --predictions my_system.csv --by domain --by is_code_mixed
python scripts/score.py --predictions my_system.csv --ci --json result.json
```

Every released utterance must appear in the file. A turn your system returned
nothing for belongs there as an empty string: it is then scored as a deletion,
which is what a downstream consumer experiences. Dropping it instead shrinks
your denominator and flatters the score, so the scorer refuses by default and
`--allow-missing` makes the choice explicit.

Or from Python:

```python
from indictelephony_bench import load_release, score, score_by, bootstrap_ci
from indictelephony_bench.data import align, load_predictions

release = load_release("all")                     # text only; no 1.7 GB download
df = align(release, load_predictions("my_system.csv"))

print(score(df))                                  # WER, CER, MER, KWER
print({k: v.wer for k, v in score_by(df, "language").items()})
print(bootstrap_ci(df, "wer"))                    # 95% CI over calls
```

## The five metrics

Full definitions in [`docs/METRICS.md`](docs/METRICS.md). In short:

| Metric | What it counts |
|---|---|
| **WER** | word substitutions, deletions and insertions over reference words |
| **CER** | the same edits over characters — independent of where word boundaries fall |
| **MER** | edits over *mixed* units: a Latin token is one unit, a native-script token its aksharas, so neither script dominates |
| **KWER** | the share of curator-tagged keywords the hypothesis fails to recover. One minus keyword recall, so inventing terms costs nothing |
| **S-WER** | WER recomputed after forgiving only *verified* same-meaning variants — script, spelling, split/join, filler, number form |

Two rules hold throughout, and they are the usual reason a number here differs
from one a quick script produces:

- **Rates are corpus-level** — total edits over total reference units, never the
  mean of per-utterance rates. 30% of this corpus is under two seconds, so
  averaging rows would mostly measure backchannels.
- **Intervals resample calls, not utterances.** Turns inside a call share a
  line, a codec path and two speakers; treating them as independent makes every
  interval too narrow.

## Semantic WER

S-WER is the one metric that calls a model, and it is optional:

```bash
pip install -e ".[semantic]"
cp .env.example .env          # add GEMINI_API_KEY
```

An LLM only *proposes* that two spans are the same word written differently.
A deterministic check then accepts or rejects each proposal — it never accepts a
pair spanning two Indic scripts, requires identical consonants within a script,
and accepts a number only when its value is confirmed. A rewrite is kept only if
it strictly reduces the edit count, so **S-WER can never exceed WER**. Answers
are cached by content, so two systems that made the same mistake are judged
identically and pay for one call.

Because a model proposes the pairs, this is the only number here that is not
perfectly reproducible: two runs of the judge moved a system by at most 0.48
points. WER, CER, MER and KWER involve no model at all.

## Installing

```bash
pip install -e .                  # scoring only
pip install -e ".[hub]"           # + pulling the release from the hub
pip install -e ".[semantic]"      # + semantic WER
pip install -e ".[hub,semantic,dev]"
pytest
```

Python 3.10+. Scoring is CPU-only; no model is trained or run locally.

## Reproducing the published numbers

The dataset gives you the references; it does not include the five vendors'
transcripts, so the published baselines are reproduced by re-running those
systems rather than by re-scoring stored output. See
[`docs/REPRODUCE.md`](docs/REPRODUCE.md) for what is and is not reproducible
from the public artifacts alone — we would rather state that plainly than imply
a one-command replication that does not exist.

## Citing

```bibtex
@misc{indictelephonybench2026,
  title        = {IndicTelephony-Bench: An Open ASR Benchmark for Indian Telephone Speech},
  author       = {Priyadarshi, Prasoon and Abdul Azeez, Zaheer and {ConvoZen AI}},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench}},
  note         = {25,393 utterances, 30.18 hours, nine languages, 8 kHz telephony}
}
```

## Licences

Code in this repository is Apache-2.0. The dataset is CC BY 4.0 and lives on the
[Hugging Face hub](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench);
its licence, and the consent that underpins it, are documented on the dataset card.
