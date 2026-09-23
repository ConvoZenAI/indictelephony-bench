"""Run one speech-to-text system over the benchmark and write its transcripts.

    python scripts/run_inference.py --provider deepgram --audio-dir ./audio
    python scripts/run_inference.py --provider deepgram --config Tamil --out preds.csv

You need this only to reproduce a baseline from scratch; scoring transcripts you
already have needs neither keys nor this script.

The output is exactly what `scripts/score.py` reads: `utterance_id,prediction`,
plus latency and status columns for inspection. Two properties matter and are
deliberate:

* **Every eligible utterance appears, including failures.** A request that fails
  is written with an empty prediction and its error, so the row is scored as a
  deletion rather than vanishing from the denominator.
* **Resumable, and retries failures.** An existing output file is read back and
  only rows without a successful transcript are requested, so a run stopped by a
  rate limit can be restarted without paying twice for what already worked -
  while the rows that failed are tried again rather than left as holes.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from indictelephony_bench.config import get_settings  # noqa: E402
from indictelephony_bench.data import (  # noqa: E402
    eligible_rows,
    load_release,
    load_release_local,
)
from indictelephony_bench.providers import get_provider  # noqa: E402
from indictelephony_bench.providers.base import SUCCESS, RowContext  # noqa: E402

FIELDS = ["utterance_id", "prediction", "latency_ms", "status", "error"]


def _done(path: Path) -> tuple[set[str], int]:
    """(utterances already transcribed, rows that failed and should be retried).

    Only a successful row counts as done. Treating a failure as done would make
    "re-run to retry the failures" quietly do nothing, and a transient rate limit
    would become a permanent hole in the run.

    A retried row is appended rather than rewritten, so the file can hold both
    attempts; the scorer is given the successful one by `load_predictions`, which
    keeps the last row for an id.
    """
    if not path.is_file():
        return set(), 0
    ok: set[str] = set()
    failed: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            uid = row.get("utterance_id")
            if not uid:
                continue
            (ok if row.get("status") == SUCCESS else failed).add(uid)
    return ok, len(failed - ok)


async def _run(rows, provider_name: str, audio_dir: Path, out: Path,
               concurrency: int) -> tuple[int, int]:
    provider = get_provider(provider_name)
    semaphore = asyncio.Semaphore(concurrency)
    new = out.stat().st_size == 0 if out.exists() else True
    ok = failed = 0
    started = time.time()

    with out.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            writer.writeheader()

        async def one(row) -> None:
            nonlocal ok, failed
            audio = audio_dir / f"{row.utterance_id}.wav"
            async with semaphore:
                if not audio.is_file():
                    result = {"prediction": "", "latency_ms": 0.0,
                              "status": "missing_audio", "error": f"{audio} not found"}
                else:
                    try:
                        result = await provider.transcribe(
                            str(audio), language=row.language_tag or None,
                            row=RowContext(language_tag=row.language_tag,
                                           base_language=row.language,
                                           is_code_mix=bool(row.is_code_mixed)))
                    except Exception as exc:  # noqa: BLE001 - one row must not end the run
                        result = {"prediction": "", "latency_ms": 0.0,
                                  "status": "error", "error": str(exc)}
            if result.get("status") == SUCCESS:
                ok += 1
            else:
                failed += 1
            writer.writerow({"utterance_id": row.utterance_id,
                             "prediction": result.get("prediction", ""),
                             "latency_ms": round(float(result.get("latency_ms") or 0), 1),
                             "status": result.get("status", ""),
                             "error": (result.get("error") or "")[:300]})
            fh.flush()          # a killed run keeps every transcript already paid for
            done = ok + failed
            if done % 50 == 0 or done == len(rows):
                rate = done / max(1e-9, time.time() - started)
                print(f"  {done:,}/{len(rows):,}  ok {ok:,}  failed {failed:,}  "
                      f"{rate:.1f}/s", flush=True)

        await asyncio.gather(*(one(r) for r in rows))
    return ok, failed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", required=True, help="registered provider id")
    ap.add_argument("--audio-dir", required=True,
                    help="folder of <utterance_id>.wav, e.g. exported from the dataset")
    ap.add_argument("--config", default="all", help="'all' or a language name")
    ap.add_argument("--local-dataset", help="use a local copy of the dataset folder")
    ap.add_argument("--out", help="output CSV (default: <provider>_predictions.csv)")
    ap.add_argument("--concurrency", type=int, default=None)
    ap.add_argument("--limit", type=int, help="stop after this many rows, for a smoke run")
    args = ap.parse_args()

    release = (load_release_local(args.local_dataset, args.config) if args.local_dataset
               else load_release(args.config))
    rows = eligible_rows(release, args.provider)
    if len(rows) != len(release):
        print(f"eligibility ({args.provider}): {len(rows):,} of {len(release):,} utterances")

    out = Path(args.out or f"{args.provider}_predictions.csv")
    already, retrying = _done(out)
    if already or retrying:
        rows = rows[~rows.utterance_id.isin(already)]
        print(f"resuming {out}: {len(already):,} already transcribed"
              + (f", {retrying:,} failed rows will be retried" if retrying else ""))
    if args.limit:
        rows = rows.head(args.limit)
    if rows.empty:
        print("nothing to do")
        return

    concurrency = args.concurrency or get_settings().max_concurrency
    print(f"{args.provider}: {len(rows):,} utterances, concurrency {concurrency}")
    ok, failed = asyncio.run(_run(list(rows.itertuples(index=False)), args.provider,
                                  Path(args.audio_dir), out, concurrency))
    print(f"\nwrote {out}  ({ok:,} ok, {failed:,} failed)")
    if failed:
        print("Failed rows are written with an empty prediction and will be scored as "
              "deletions. Re-run this command to retry only those rows.")


if __name__ == "__main__":
    main()
