# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the versions track
the dataset release they score.

## [Unreleased]

No published number changes: scoring is untouched.

### Changed
- Akshara is re-run through the public ConvoZen SDK
  ([`convozen` on PyPI](https://pypi.org/project/convozen/),
  key in `CONVOZEN_API_KEY`), like every other system. As in the published run,
  no language hint is sent (`AKSHARA_LANGUAGE_HINTS=true` sends the utterance's
  languages), and the reference keywords are never sent.

### Fixed
- Providers report success as `"success"`, but the runner and the prediction
  loader checked for `"ok"`. Every successful row was counted as failed and a
  resumed run re-requested all of them. One `SUCCESS` constant is now shared.
- `pip install indictelephony-bench[akshara]` no longer needs `httpx`.

## [1.0.0] - 2026-09-21

First public release. Scores the IndicTelephony-Bench dataset: 25,393
utterances, 30.18 hours, nine languages, 8 kHz telephone audio.
