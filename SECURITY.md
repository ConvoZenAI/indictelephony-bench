# Security policy

## Reporting a vulnerability

Please report security issues privately rather than opening a public issue.
Use GitHub's **Report a vulnerability** button on the Security tab, which opens
a private advisory.

We aim to acknowledge a report within five working days.

## Scope

This repository scores text. It trains no model, serves no network endpoint and
executes nothing from the dataset, so the realistic surface is narrow:

- **Credentials.** No key is ever read from source. Keys come from the
  environment or a local `.env`, which is gitignored. If you find a credential
  committed anywhere in this history, report it as a vulnerability.
- **Untrusted input.** Transcripts and predictions are parsed as text, never
  evaluated. A crafted CSV should produce an error, not an execution; if it does
  otherwise, that is a bug worth reporting.
- **Dependencies.** Runtime dependencies are pinned by lower bound in
  `pyproject.toml` and kept deliberately few.

## Data

The dataset is not distributed from this repository. Questions about consent,
licensing or personal data in the recordings belong on the
[dataset card](https://huggingface.co/datasets/ConvoZenAI/indictelephony-bench).
