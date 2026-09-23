# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions track
the dataset release they score.

## [Unreleased]

No published number changes: scoring is untouched.

### Changed
- Akshara is re-run through the public ConvoZen SDK
  ([`convozen` on PyPI](https://pypi.org/project/convozen/),
  key in `CONVOZEN_API_KEY`), like every other system. Language hints are
  taken from each utterance's tag, and the reference keywords are never sent.

### Fixed
- Providers report success as `"success"`, but the runner and the prediction
  loader checked for `"ok"`. Every successful row was counted as failed and a
  resumed run re-requested all of them. One `SUCCESS` constant is now shared.
- `pip install indictelephony-bench[akshara]` no longer needs `httpx`.

## [1.0.1] - 2026-09-21

Scores release **v1.0.1** of the dataset: 25,393 utterances, 30.18 hours, 320 calls.

### Changed
- Automotive utterances of 1.0-2.0 s were removed from Telugu, Kannada,
  Malayalam, Hindi, Gujarati and Bengali to reduce a domain and duration skew.
  Tamil and English are untouched: removing anything would take them below the
  three-hour-per-language minimum the release claims. WER moves between -0.26
  and -0.55 points; no ranking changes on any metric.

## [1.0.0] - 2026-09-16

First frozen release: 28,111 utterances, 31.31 hours, 320 calls. Retained
unchanged and checksum-frozen; v1.0.1 is a new freeze rather than an edit, so
numbers computed against either version stay reproducible.
