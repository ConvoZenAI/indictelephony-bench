## What this changes

## Does it move a published number?

- [ ] No — refactor, docs, tests or a new optional metric
- [ ] Yes — and the PR adds a `CHANGELOG.md` entry saying which numbers moved
      and by how much, and bumps the version

A change that moves a published score lands as a new version rather than an
edit; see [CONTRIBUTING.md](../CONTRIBUTING.md) for why.

## Checks

- [ ] `pytest` passes
- [ ] New behaviour in `metrics/` or `normalize.py` has a test
- [ ] No new required dependency (or the PR says why one is needed)
