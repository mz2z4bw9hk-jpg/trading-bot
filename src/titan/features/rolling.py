"""Vectorized causal rolling primitives shared by feature builders.

Every function here is strictly causal: the value at position ``t`` depends
only on inputs at positions ``<= t``. The feature-level causality contract is
enforced by ``FeatureRegistry.verify_causality`` and the test suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_slope_stats(y: pd.Series, window: int) -> tuple[pd.Series, pd.Series]:
    """Rolling OLS of ``y`` on time. Returns (slope, t_stat) aligned to ``y``.

    Closed-form via sliding windows; O(n·w) with vectorized matmul. The t-stat
    of the slope is a scale-free trend-strength measure (Student-t under the
    i.i.d. residual null), preferable to raw slope because it is comparable
    across instruments and volatility levels.
    """
    n = len(y)
    slope = np.full(n, np.nan)
    tstat = np.full(n, np.nan)
    if n < window or window < 3:
        return pd.Series(slope, index=y.index), pd.Series(tstat, index=y.index)

    values = y.to_numpy(dtype=float)
    x = np.arange(window, dtype=float)
    sx, sxx = x.sum(), (x * x).sum()
    denom_x = window * sxx - sx * sx

    win = np.lib.stride_tricks.sliding_window_view(values, window)
    sy = win.sum(axis=1)
    syy = (win * win).sum(axis=1)
    sxy = win @ x

    num = window * sxy - sx * sy
    slope_v = num / denom_x
    denom_y = window * syy - sy * sy
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / np.sqrt(denom_x * np.clip(denom_y, 0, None))
        r = np.clip(r, -0.999999, 0.999999)
        t = r * np.sqrt((window - 2) / (1.0 - r * r))

    slope[window - 1 :] = slope_v
    tstat[window - 1 :] = t
    return pd.Series(slope, index=y.index), pd.Series(tstat, index=y.index)


def efficiency_ratio(close: pd.Series, window: int) -> pd.Series:
    """Kaufman efficiency ratio: net move / path length, in [0, 1]."""
    net = (close - close.shift(window)).abs()
    path = close.diff().abs().rolling(window).sum()
    return (net / path.replace(0.0, np.nan)).fillna(0.0)


def wilder_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """RSI with Wilder's smoothing (EWMA alpha=1/window), strictly causal."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()
    rsi = 100.0 - 100.0 / (1.0 + gain / loss.replace(0.0, np.nan))
    # Zero average loss is not "neutral": all-gain windows are RSI 100.
    monotone_up = (loss == 0.0) & (gain > 0.0)
    rsi = rsi.mask(monotone_up, 100.0)
    return rsi.fillna(50.0)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average true range (Wilder smoothing)."""
    return true_range(df).ewm(alpha=1.0 / window, adjust=False, min_periods=window).mean()


def realized_vol(close: pd.Series, window: int, annualize: float = 252.0) -> pd.Series:
    return np.log(close).diff().rolling(window).std() * np.sqrt(annualize)


def parkinson_vol(df: pd.DataFrame, window: int, annualize: float = 252.0) -> pd.Series:
    """Parkinson range estimator: more efficient than close-to-close."""
    hl = np.log(df["high"] / df["low"]) ** 2
    return np.sqrt(hl.rolling(window).mean() / (4.0 * np.log(2.0))) * np.sqrt(annualize)


def garman_klass_vol(df: pd.DataFrame, window: int, annualize: float = 252.0) -> pd.Series:
    """Garman-Klass OHLC estimator."""
    hl = 0.5 * np.log(df["high"] / df["low"]) ** 2
    co = (2.0 * np.log(2.0) - 1.0) * np.log(df["close"] / df["open"]) ** 2
    var = (hl - co).rolling(window).mean().clip(lower=0.0)
    return np.sqrt(var) * np.sqrt(annualize)


def on_balance_volume(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff()).fillna(0.0)
    return (direction * df["volume"]).cumsum()


def rolling_percentile_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile of the current value within its own trailing window, in [0,1]."""

    def _rank(win: np.ndarray) -> float:
        return float((win[:-1] <= win[-1]).mean()) if len(win) > 1 else np.nan

    return s.rolling(window, min_periods=max(10, window // 4)).apply(_rank, raw=True)


def signed_streak(close: pd.Series, cap: int = 10) -> pd.Series:
    """Length of the current run of same-sign daily moves, signed, capped."""
    sign = np.sign(close.diff().to_numpy())
    out = np.zeros(len(sign))
    for i in range(1, len(sign)):
        if sign[i] == 0 or np.isnan(sign[i]):
            out[i] = 0.0
        elif sign[i] == np.sign(out[i - 1]) or out[i - 1] == 0:
            out[i] = out[i - 1] + sign[i] if np.sign(out[i - 1]) == sign[i] else sign[i]
        else:
            out[i] = sign[i]
    return pd.Series(np.clip(out, -cap, cap), index=close.index)
