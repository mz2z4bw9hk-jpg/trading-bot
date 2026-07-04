"""Triple-barrier labelling: hand-checkable scenarios."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import LabelConfig
from titan.labels.triple_barrier import (
    average_uniqueness,
    ewma_volatility,
    triple_barrier_labels,
)


def _flat_frame(n: int, close: float = 100.0) -> pd.DataFrame:
    idx = pd.bdate_range("2021-01-01", periods=n, tz="UTC")
    return pd.DataFrame(
        {"open": close, "high": close * 1.001, "low": close * 0.999,
         "close": close, "volume": 1e6},
        index=idx,
    )


def _cfg(horizon=5, tp=2.0, sl=1.5, span=10) -> LabelConfig:
    return LabelConfig(horizon_bars=horizon, tp_sigma=tp, sl_sigma=sl, vol_span=span)


def test_tp_touch_wins():
    cfg = _cfg()
    df = _flat_frame(60)
    # inject noise so sigma > floor, then a clean TP touch at t+2
    rng = np.random.default_rng(0)
    noise = 1 + 0.01 * rng.standard_normal(len(df))
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * noise
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999

    t = 40
    sigma = ewma_volatility(df["close"], cfg.vol_span).iloc[t]
    entry = df["open"].iloc[t + 1]
    tp_price = entry * np.exp(cfg.tp_sigma * sigma)
    df.iloc[t + 2, df.columns.get_loc("high")] = tp_price * 1.001

    ls = triple_barrier_labels(df, cfg).frame
    row = ls.loc[df.index[t]]
    assert row["label"] == 1
    assert row["touch"] == "tp"
    assert row["bars_held"] == 2
    assert row["t1"] == df.index[t + 2]


def test_sl_touch_and_pessimistic_tie():
    cfg = _cfg()
    df = _flat_frame(60)
    rng = np.random.default_rng(1)
    noise = 1 + 0.01 * rng.standard_normal(len(df))
    for col in ("open", "high", "low", "close"):
        df[col] = df[col] * noise
    df["high"] = df[["open", "close"]].max(axis=1) * 1.001
    df["low"] = df[["open", "close"]].min(axis=1) * 0.999

    t = 30
    sigma = ewma_volatility(df["close"], cfg.vol_span).iloc[t]
    entry = df["open"].iloc[t + 1]
    # both barriers pierced on the SAME bar -> stop must win (pessimistic)
    df.iloc[t + 1, df.columns.get_loc("high")] = entry * np.exp(cfg.tp_sigma * sigma) * 1.01
    df.iloc[t + 1, df.columns.get_loc("low")] = entry * np.exp(-cfg.sl_sigma * sigma) * 0.99

    ls = triple_barrier_labels(df, cfg).frame
    row = ls.loc[df.index[t]]
    assert row["label"] == 0
    assert row["touch"] == "sl"
    assert row["ret"] == pytest.approx(-cfg.sl_sigma * sigma)


def test_time_exit_uses_terminal_sign(ohlcv):
    cfg = _cfg(horizon=8)
    ls = triple_barrier_labels(ohlcv, cfg).frame
    time_rows = ls[ls["touch"] == "time"]
    assert len(time_rows) > 0
    closes = ohlcv["close"]
    for ts, row in time_rows.head(20).iterrows():
        pos = ohlcv.index.get_loc(ts)
        terminal = closes.iloc[pos + cfg.horizon_bars]
        assert row["label"] == int(terminal > row["entry"])


def test_mae_mfe_bounds(ohlcv):
    ls = triple_barrier_labels(ohlcv, _cfg(horizon=10)).frame
    assert (ls["mae"] <= 1e-9).all()          # adverse excursion can't be positive
    assert (ls["mfe"] >= -1e-9).all()
    assert (ls["mae"] <= ls["mfe"]).all()


def test_average_uniqueness_overlap():
    calendar = pd.bdate_range("2021-01-01", periods=30, tz="UTC")
    # two fully overlapping events + one isolated
    starts = pd.DatetimeIndex([calendar[0], calendar[0], calendar[20]])
    ends = pd.DatetimeIndex([calendar[5], calendar[5], calendar[25]])
    u = average_uniqueness(starts, ends, calendar)
    assert u.iloc[0] == pytest.approx(0.5)
    assert u.iloc[2] == pytest.approx(1.0)
