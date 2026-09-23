# Contributing

Thanks for looking. This repository is the scoring side of a published
benchmark, which shapes what changes are easy to accept.

## The one rule that matters

**A change that moves a published number needs a new version, not an edit.**

The numbers in the paper and the article were computed with a specific
normalizer and specific metric definitions. Quietly improving either would make
every published table wrong in a way nobody can detect from the output. So:

- Bug fixes that change scores are welcome, but they land as a new minor version
  with a `CHANGELOG.md` entry stating what moved and by how much.
- Adding an equivalence to the normalizer's list, changing a bucket edge or
  loosening the semantic check all count as changing scores.
- Refactoring, documentation, tests and new *optional* metrics are free.

## Setting up

```bash
git clone https://github.com/ConvoZenAI/indictelephony-bench
cd indictelephony-bench
pip install -e ".[hub,semantic,dev]"
pytest
```

## What a good change looks like

- **Tests first for anything in `metrics/` or `normalize.py`.** These are small,
  pure functions; a test that states the intended behaviour in one line is worth
  more than a paragraph of review. `tests/` has 135 of them to copy the style from.
- **Explain the why, not the what, in comments.** The code says what it does. A
  comment earns its place by recording the reasoning a future reader would
  otherwise have to reconstruct — usually a trade-off, a failed alternative or a
  rule that is easy to violate by accident.
- **Keep corpus-level aggregation corpus-level.** If a change introduces a mean
  of per-row rates anywhere, it is almost certainly wrong; see `docs/METRICS.md`.
- **No new required dependency** without a reason in the PR description. The
  deterministic path is deliberately installable with four packages.

## Running the checks

```bash
pytest                    # 135 tests, under a second
python -m compileall src  # syntax
```

CI runs the same on 3.10, 3.11 and 3.12.

## Reporting a scoring disagreement

If your own scorer disagrees with this one, that is a useful bug report. Include
the reference, the hypothesis and both numbers. Most disagreements turn out to
be one of three things, all documented in `docs/METRICS.md`: a mean of row rates
instead of a corpus rate, a different normalizer, or utterances dropped from the
denominator rather than scored as deletions.

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).
