"""Aggregate OHLCV bars to a coarser interval.

Two problems this solves, both of which otherwise end a run before it starts:

1. **Intervals the vendor does not serve.** Yahoo has no native 2h/3h/4h bar,
   so those timeframes are declarable but unobtainable without aggregation.

2. **Volume the vendor only populates sparsely.** Yahoo's hourly crypto series
   carries no volume on roughly half its bars. Summing N source bars into one
   target bar makes the target zero-volume only where *every* source bar in it
   was empty, so aggregating trades resolution for a complete volume column —
   the one the OBV, dollar-volume and up/down-volume features are built from.
   This recovers real traded volume; it does not invent any.

CAUSALITY. Bars are stamped at the START of the interval they cover
(``label="left"``, ``closed="left"``), matching the convention the rest of the
platform assumes: a bar stamped ``t`` is complete at ``t + delta``, and the
engine enters at the open of the following bar. Stamping right would date a
bar before some of the data inside it and leak the future into every feature.
"""

from __future__ import annotations

import pandas as pd

from titan.core.log import get_logger

logger = get_logger(__name__)

_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

# pandas offset alias per timeframe, for the coarser (target) side.
RESAMPLE_RULES: dict[str, str] = {
    "1wk": "1W",
    "1d": "1D",
    "4h": "4h",
    "3h": "3h",
    "2h": "2h",
    "1h": "1h",
    "30m": "30min",
    "15m": "15min",
    "5m": "5min",
    "1m": "1min",
}


def _aggregate(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    cols = [c for c in _AGG if c in df.columns]
    out = df.resample(rule, label="left", closed="left").agg(
        {c: _AGG[c] for c in cols}
    )
    # Buckets covering a market closure have no source rows at all.
    return out.dropna(subset=["close"])


def resample_ohlcv(df: pd.DataFrame, rule: str, *, within_sessions: bool) -> pd.DataFrame:
    """Aggregate to ``rule``; never merge across a session boundary.

    On a market that closes, a naive resample would fuse the last bars of one
    session with the first of the next into a single bar spanning the
    overnight gap — a bar that never traded as one. Session markets are
    therefore aggregated per calendar day and re-stitched.
    """
    if df.empty:
        return df
    if not within_sessions:
        return _aggregate(df, rule)

    days = [_aggregate(day, rule) for _, day in df.groupby(df.index.normalize())]
    frames = [d for d in days if not d.empty]
    if not frames:
        return df.iloc[:0]
    return pd.concat(frames).sort_index()


def zero_volume_fraction(df: pd.DataFrame) -> float:
    if "volume" not in df.columns or df.empty:
        return 0.0
    return float((df["volume"] <= 0).mean())
