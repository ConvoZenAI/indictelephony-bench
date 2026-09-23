"""Corpus-level scoring, and call-level confidence intervals.

Two rules hold everywhere in this module, and they are the reason a number here
can differ from one a quick script produces:

* A corpus rate is total edits over total reference units, never the mean of
  per-utterance rates. Averaging rows lets a one-word backchannel weigh as much
  as a twenty-word sentence, and a benchmark that is 30% backchannels would then
  mostly measure backchannels.
* Resampling is over calls, not utterances. Turns inside one call share a line,
  a codec path and two speakers, so treating them as independent draws makes
  every interval too narrow.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from indictelephony_bench.metrics.cer import compute_cer_counts
from indictelephony_bench.metrics.keywords import compute_keyword_counts
from indictelephony_bench.metrics.mer import compute_mer_counts
from indictelephony_bench.metrics.wer import compute_wer_counts

SEED = 20260916
N_BOOT = 1000


@dataclass(frozen=True)
class Score:
    """One system on one set of rows. Rates are fractions, not percentages."""

    utterances: int
    wer: float
    cer: float
    mer: float
    kwer: float | None
    reference_words: int
    keywords_scored: int
    empty_hypotheses: int

    def as_dict(self) -> dict:
        return asdict(self)


def _counts(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row edit and unit counts. Summing these columns gives a corpus rate.

    Kept as counts rather than rates precisely so that the aggregation stays a
    sum over units; a rate column would invite someone to take its mean.
    """
    rows = []
    for r in df.itertuples(index=False):
        ref, hyp = r.transcription, r.prediction
        _, w_edits, w_units = compute_wer_counts(ref, hyp)
        _, c_edits, c_units = compute_cer_counts(ref, hyp)
        _, m_edits, m_units = compute_mer_counts(ref, hyp)
        kw_hits, kw_total = compute_keyword_counts(r.keywords, hyp, reference=ref)
        rows.append((w_edits, w_units, c_edits, c_units, m_edits, m_units,
                     kw_hits, kw_total, int(not str(hyp).strip())))
    return pd.DataFrame(rows, index=df.index, columns=[
        "w_edits", "w_units", "c_edits", "c_units", "m_edits", "m_units",
        "kw_hits", "kw_total", "empty_hyp"])


def _rate(edits: float, units: float) -> float:
    return float(edits) / float(units) if units else float("nan")


def score(df: pd.DataFrame, *, counts: pd.DataFrame | None = None) -> Score:
    """Every deterministic metric over the given rows, at corpus level."""
    c = _counts(df) if counts is None else counts
    total = c.sum()
    return Score(
        utterances=int(len(df)),
        wer=_rate(total["w_edits"], total["w_units"]),
        cer=_rate(total["c_edits"], total["c_units"]),
        mer=_rate(total["m_edits"], total["m_units"]),
        kwer=(1.0 - total["kw_hits"] / total["kw_total"]) if total["kw_total"] else None,
        reference_words=int(total["w_units"]),
        keywords_scored=int(total["kw_total"]),
        empty_hypotheses=int(total["empty_hyp"]),
    )


def score_by(df: pd.DataFrame, column: str) -> dict[object, Score]:
    """The same scores split by any column: language, domain, is_code_mixed.

    Each group is scored at corpus level within itself, so the groups do not
    average to the overall rate unless the groups happen to be equal in size.
    """
    counts = _counts(df)
    return {key: score(part, counts=counts.loc[part.index])
            for key, part in df.groupby(column, sort=True)}


def bootstrap_ci(df: pd.DataFrame, metric: str = "wer", *, n_boot: int = N_BOOT,
                 seed: int = SEED, alpha: float = 0.05) -> tuple[float, float]:
    """A percentile interval for one metric, resampling calls with replacement.

    The same seed and the same call order give the same interval, so a published
    interval is reproducible rather than merely plausible.
    """
    pairs = {"wer": ("w_edits", "w_units"), "cer": ("c_edits", "c_units"),
             "mer": ("m_edits", "m_units")}
    if metric not in pairs and metric != "kwer":
        raise ValueError(f"metric must be one of {sorted(set(pairs) | {'kwer'})}")
    counts = _counts(df)
    counts["call_id"] = df.call_id.values
    per_call = counts.groupby("call_id", sort=True).sum()
    rng = np.random.default_rng(seed)
    calls = per_call.index.to_numpy()
    draws = rng.integers(0, len(calls), size=(n_boot, len(calls)))

    if metric == "kwer":
        hits, totals = per_call["kw_hits"].to_numpy(), per_call["kw_total"].to_numpy()
        vals = np.array([1.0 - hits[d].sum() / totals[d].sum() if totals[d].sum() else np.nan
                         for d in draws])
    else:
        e_col, u_col = pairs[metric]
        edits, units = per_call[e_col].to_numpy(), per_call[u_col].to_numpy()
        vals = np.array([edits[d].sum() / units[d].sum() if units[d].sum() else np.nan
                         for d in draws])
    if not np.isfinite(vals).any():
        # Nothing to resample - a KWER interval over rows with no scorable
        # keyword, say. NaN is the honest answer; numpy would also print an
        # "All-NaN slice" warning, which is not the caller's problem.
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def paired_delta_ci(df_a: pd.DataFrame, df_b: pd.DataFrame, metric: str = "wer", *,
                    n_boot: int = N_BOOT, seed: int = SEED,
                    alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """Interval for the gap between two systems, scored on the same resamples.

    Two independent intervals can overlap while the systems are still reliably
    apart, because the same hard calls hurt both. Resampling both systems on one
    draw of calls answers the question actually being asked: is A better than B
    on this corpus?
    """
    if metric not in {"wer", "cer", "mer"}:
        raise ValueError("paired intervals are defined here for wer, cer and mer")
    pairs = {"wer": ("w_edits", "w_units"), "cer": ("c_edits", "c_units"),
             "mer": ("m_edits", "m_units")}
    e_col, u_col = pairs[metric]

    def per_call(df: pd.DataFrame) -> pd.DataFrame:
        c = _counts(df)
        c["call_id"] = df.call_id.values
        return c.groupby("call_id", sort=True).sum()

    a, b = per_call(df_a), per_call(df_b)
    shared = a.index.intersection(b.index)
    if len(shared) < len(a.index) or len(shared) < len(b.index):
        a, b = a.loc[shared], b.loc[shared]
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(shared), size=(n_boot, len(shared)))
    ae, au = a[e_col].to_numpy(), a[u_col].to_numpy()
    be, bu = b[e_col].to_numpy(), b[u_col].to_numpy()
    deltas = np.array([be[d].sum() / bu[d].sum() - ae[d].sum() / au[d].sum() for d in draws])
    point = be.sum() / bu.sum() - ae.sum() / au.sum()
    lo, hi = np.nanpercentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), (float(lo), float(hi))
