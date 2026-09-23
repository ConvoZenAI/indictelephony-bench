"""Loading the released benchmark, and the predictions you score against it.

The release lives on the Hugging Face hub, one parquet config per language plus
an `all` config. Nothing here needs the audio to score text, so `load_release`
skips the audio column by default: it turns a 1.7 GB download into a few MB.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import pandas as pd

DATASET_ID = "ConvoZenAI/indictelephony-bench"

# Statuses that mean a row has a real transcript. "success" is what the
# runner writes; "ok" is accepted for prediction files written by hand.
SUCCESS_STATUSES = frozenset({"success", "ok"})

# config name -> the `language` value carried by its rows
LANGUAGES = {
    "Bengali": "bn", "English": "en", "Gujarati": "gu", "Hindi": "hi",
    "Kannada": "kn", "Malayalam": "ml", "Marathi": "mr", "Tamil": "ta",
    "Telugu": "te",
}

# What a release row carries. `transcription` is the human reference;
# `transcription_normalized` is that reference after the scoring normalizer,
# so a scorer can be checked without reimplementing normalization.
RELEASE_COLUMNS = [
    "utterance_id", "call_id", "language", "language_tag", "is_code_mixed",
    "duration_sec", "domain", "transcription", "transcription_normalized", "keywords",
]


def load_release(config: str = "all", *, with_audio: bool = False,
                 revision: str | None = None) -> pd.DataFrame:
    """The released references for one language, or for all nine.

    `config` is a language name from LANGUAGES, or "all". With `with_audio`,
    the audio column comes too, as {"bytes": ..., "path": ...} per row; without
    it only the text columns are fetched, which is all scoring needs.
    """
    if config != "all" and config not in LANGUAGES:
        raise ValueError(f"unknown config {config!r}; expected 'all' or one of {sorted(LANGUAGES)}")
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, config, split="test", revision=revision)
    if not with_audio:
        ds = ds.remove_columns([c for c in ds.column_names if c == "audio"])
    return ds.to_pandas()


def load_release_local(folder: str | Path, config: str = "all") -> pd.DataFrame:
    """The same rows from a local copy of the dataset folder, for offline use.

    Expects the published layout: <folder>/<Language>/test-*.parquet.
    """
    root = Path(folder)
    globs = sorted(root.glob("*/test-*.parquet") if config == "all"
                   else root.glob(f"{config}/test-*.parquet"))
    if not globs:
        raise FileNotFoundError(f"no parquet shards under {root} for config {config!r}")
    import pyarrow.parquet as pq

    keep = RELEASE_COLUMNS
    return pd.concat([pq.read_table(g, columns=keep).to_pandas() for g in globs],
                     ignore_index=True)


def load_predictions(path: str | Path, *, id_column: str = "utterance_id",
                     text_column: str = "prediction") -> pd.Series:
    """One system's output, as utterance_id -> hypothesis.

    A CSV or parquet with an id column and a text column. Missing text is read
    as an empty hypothesis, which is how a turn a system returned nothing for is
    scored: fully deleted, never skipped.
    """
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, dtype=str, keep_default_na=False)
    for col in (id_column, text_column):
        if col not in df.columns:
            raise KeyError(f"{p} has no column {col!r}; found {list(df.columns)}")
    if "status" in df.columns and df[id_column].duplicated().any():
        # A runner appends a retry rather than rewriting the failed attempt, so
        # an id can legitimately appear twice. The successful attempt is the
        # transcript the system produced; prefer it, and otherwise take the last.
        df = df.assign(_ok=df["status"].str.lower().isin(SUCCESS_STATUSES).astype(int))
        df = df.sort_values("_ok", kind="stable").drop_duplicates(id_column, keep="last")
    s = df.set_index(id_column)[text_column].fillna("")
    if s.index.duplicated().any():
        dupes = s.index[s.index.duplicated()].unique()[:3]
        raise ValueError(
            f"{p} repeats {int(s.index.duplicated().sum())} utterance ids, e.g. {list(dupes)}. "
            f"Scoring one id twice would weight it twice; de-duplicate first, or add a "
            f"`status` column so the successful attempt can be chosen.")
    return s


def align(release: pd.DataFrame, predictions: pd.Series, *,
          require_complete: bool = True) -> pd.DataFrame:
    """Join predictions onto the release, one row per scored utterance.

    With `require_complete`, a missing utterance is an error rather than a
    silent omission: a system scored on fewer rows than another is not comparable
    to it, and that is exactly the kind of gap a leaderboard hides.
    """
    df = release.copy()
    df["prediction"] = df.utterance_id.map(predictions)
    missing = df.prediction.isna()
    if missing.any():
        if require_complete:
            raise ValueError(
                f"{int(missing.sum())} of {len(df)} released utterances have no prediction, "
                f"e.g. {list(df.utterance_id[missing][:3])}. Pass require_complete=False to "
                f"score them as empty hypotheses instead.")
        df.loc[missing, "prediction"] = ""
    return df


def eligible_rows(df: pd.DataFrame, provider: str) -> pd.DataFrame:
    """The rows a provider should be scored on, dropping languages it cannot do.

    Scoring a system on a language it does not support measures the gap in its
    language list, not its recognition, so those rows leave its denominator.
    """
    from indictelephony_bench.eligibility import is_row_eligible

    keep: Iterable[bool] = [
        is_row_eligible(provider, r.language, r.language_tag) for r in df.itertuples(index=False)
    ]
    return df[list(keep)]
