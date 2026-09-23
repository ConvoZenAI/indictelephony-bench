"""Score one system's transcripts against IndicTelephony-Bench.

    python scripts/score.py --predictions my_system.csv
    python scripts/score.py --predictions my_system.csv --config Tamil --by language
    python scripts/score.py --predictions my_system.csv --provider deepgram --json out.json

`--predictions` is a CSV or parquet with an `utterance_id` column and a
`prediction` column. Every released utterance must appear; a turn the system
returned nothing for belongs in the file as an empty string, because scoring it
as a deletion is the honest treatment and dropping it is not.

`--provider` applies the published eligibility rule, which removes the languages
a system does not claim to support. Use it only for a system that genuinely has
no model for a language, and say so when reporting: a smaller denominator is not
a better score.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from indictelephony_bench.data import (
    align,
    eligible_rows,
    load_predictions,
    load_release,
    load_release_local,
)
from indictelephony_bench.scoring import bootstrap_ci, score, score_by


def main() -> None:
    """Entry point for `indictelephony-score` and `scripts/score.py`."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, help="CSV/parquet of utterance_id,prediction")
    ap.add_argument("--config", default="all", help="'all' or a language name, e.g. Tamil")
    ap.add_argument("--local-dataset", help="score against a local copy of the dataset folder")
    ap.add_argument("--provider", help="apply the eligibility rule for this provider id")
    ap.add_argument("--by", action="append", default=[],
                    help="also report a breakdown by this column; repeatable "
                         "(language, domain, is_code_mixed)")
    ap.add_argument("--ci", action="store_true", help="add a 95%% call-level bootstrap interval for WER")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--allow-missing", action="store_true",
                    help="score absent utterances as empty rather than failing")
    ap.add_argument("--json", help="write the full result here")
    args = ap.parse_args()

    try:
        release = (load_release_local(args.local_dataset, args.config) if args.local_dataset
                   else load_release(args.config))
        preds = load_predictions(args.predictions)
        df = align(release, preds, require_complete=not args.allow_missing)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        # These are the mistakes a first run actually makes: the wrong path, the
        # wrong column name, a predictions file that is missing rows. Each message
        # already says what to do, so print it and stop rather than a traceback.
        sys.exit(f"error: {exc}")
    if args.provider:
        before = len(df)
        df = eligible_rows(df, args.provider)
        if len(df) != before:
            print(f"eligibility ({args.provider}): scoring {len(df):,} of {before:,} utterances")

    overall = score(df)
    result = {"predictions": args.predictions, "config": args.config,
              "provider": args.provider, "overall": overall.as_dict()}

    print(f"\n{'':<22}{'WER':>8}{'CER':>8}{'MER':>8}{'KWER':>8}{'utts':>9}")
    def row(label: str, s) -> None:
        kw = f"{100 * s.kwer:7.2f}%" if s.kwer is not None else f"{'-':>8}"
        print(f"{label:<22}{100 * s.wer:7.2f}%{100 * s.cer:7.2f}%{100 * s.mer:7.2f}%{kw}{s.utterances:9,}")
    row("overall", overall)

    if args.ci:
        lo, hi = bootstrap_ci(df, "wer", n_boot=args.n_boot)
        result["overall"]["wer_ci"] = [lo, hi]
        print(f"{'  95% CI (WER)':<22}{100 * lo:7.2f}% - {100 * hi:.2f}%   "
              f"({args.n_boot:,} resamples over {df.call_id.nunique()} calls)")

    for column in args.by:
        if column not in df.columns:
            sys.exit(f"--by {column}: no such column; have {sorted(df.columns)}")
        print()
        parts = score_by(df, column)
        result[f"by_{column}"] = {str(k): v.as_dict() for k, v in parts.items()}
        for key, s in parts.items():
            row(f"  {column}={key}", s)

    if overall.empty_hypotheses:
        print(f"\n{overall.empty_hypotheses:,} utterances have an empty hypothesis; "
              f"each is scored as fully deleted.")

    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
