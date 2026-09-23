"""End-to-end check of the scorer on a tiny fixture: no network, no dataset.

    python scripts/selftest.py

CI runs this so that a broken entry point fails here rather than for a user.
It also doubles as the shortest readable example of the file formats.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

RELEASE = """utterance_id,call_id,language,language_tag,is_code_mixed,duration_sec,domain,transcription,transcription_normalized,keywords
u1,c1,hi,en-hi,True,4.1,banking_finance,mera loan amount twenty six thousand hai,mera loan amount 2 6 0 0 0 hai,"loan, twenty six thousand"
u2,c1,hi,hi,False,0.9,other,haan ji,haan ji,
u3,c2,en,en,False,3.2,insurance,my policy number please,my policy number please,policy number
u4,c2,en,en,False,1.1,other,ok thank you,ok thank you,
"""

PREDICTIONS = """utterance_id,prediction
u1,mera loan amount 26000 hai
u2,haan ji
u3,my policy no please
u4,
"""


def main() -> int:
    import pandas as pd

    from indictelephony_bench.data import align, load_predictions
    from indictelephony_bench.scoring import score, score_by

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "release.csv").write_text(RELEASE)
        (tmp / "preds.csv").write_text(PREDICTIONS)

        release = pd.read_csv(tmp / "release.csv")
        df = align(release, load_predictions(tmp / "preds.csv"))
        result = score(df)

        checks: list[tuple[str, bool, object]] = [
            ("all four utterances scored", result.utterances == 4, result.utterances),
            # "twenty six thousand" and "26000" normalize to the same digits, so a
            # system that prints digits is not charged for house style.
            ("number form forgiven", result.wer < 0.30, round(result.wer, 4)),
            # u4 returned nothing: scored as a deletion, never skipped.
            ("empty hypothesis counted", result.empty_hypotheses == 1, result.empty_hypotheses),
            # 3 keywords in the references, 2 recovered.
            ("keywords scored", result.keywords_scored == 3, result.keywords_scored),
            ("kwer is one minus recall", abs(result.kwer - 1 / 3) < 1e-9, round(result.kwer, 4)),
            ("grouping works", set(score_by(df, "language")) == {"en", "hi"},
             sorted(score_by(df, "language"))),
        ]
        ok = True
        for name, passed, got in checks:
            print(f"  [{'ok ' if passed else 'FAIL'}] {name}  ({got})")
            ok &= passed

        # the CLI itself, not just the library
        proc = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "score.py"),
             "--predictions", str(tmp / "preds.csv"),
             "--local-dataset", str(tmp), "--allow-missing"],
            capture_output=True, text=True)
        cli_ok = proc.returncode != 0 and "no parquet shards" in (proc.stderr + proc.stdout)
        print(f"  [{'ok ' if cli_ok else 'FAIL'}] CLI reports a missing dataset clearly")
        ok &= cli_ok

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
