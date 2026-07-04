"""Canonical OHLCV schema enforcement.

Every provider's output passes through :func:`normalize_ohlcv` before anything
downstream sees it, so features, labels and the backtester can assume a single
well-formed representation: UTC ``DatetimeIndex``, sorted, unique, float64
columns ``open/high/low/close/volume``.
"""

from __future__ import annotations

import pandas as pd

from titan.core.types import OHLCV_COLUMNS


class SchemaError(ValueError):
    """Raised when a frame cannot be coerced to the canonical OHLCV schema."""


def normalize_ohlcv(df: pd.DataFrame, max_forward_fill: int = 2) -> pd.DataFrame:
    """Coerce an arbitrary provider frame to the canonical schema.

    Parameters
    ----------
    df:
        Raw frame with (case-insensitive) open/high/low/close/volume columns
        and a datetime-like index.
    max_forward_fill:
        Maximum number of consecutive missing bars to forward-fill. Anything
        longer is left as NaN and later penalised by quality scoring.
    """
    if df is None or len(df) == 0:
        raise SchemaError("empty frame")

    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]

    # Providers sometimes give "adj close"; prefer it as close if close missing.
    if "close" not in out.columns and "adj close" in out.columns:
        out = out.rename(columns={"adj close": "close"})

    missing = [c for c in OHLCV_COLUMNS if c not in out.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")
    out = out[list(OHLCV_COLUMNS)]

    idx = pd.DatetimeIndex(pd.to_datetime(out.index))
    idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
    out.index = idx
    out = out[~out.index.duplicated(keep="last")].sort_index()

    out = out.astype("float64")
    # Drop rows where the entire bar is missing, then limited forward fill for
    # isolated gaps (prices only; a filled bar has zero volume by construction).
    out = out.dropna(how="all")
    price_cols = ["open", "high", "low", "close"]
    filled_mask = out["close"].isna()
    if max_forward_fill > 0:
        out[price_cols] = out[price_cols].ffill(limit=max_forward_fill)
        out.loc[filled_mask & out["close"].notna(), "volume"] = 0.0
    out = out.dropna(subset=["close"])

    if len(out) == 0:
        raise SchemaError("no usable rows after normalization")
    return out
