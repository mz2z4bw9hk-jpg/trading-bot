"""QC must not read a market's opening hours as missing data.

Two checks were calendar-naive. On hourly US-equity bars the overnight
boundary is every seventh bar, so gap detection scored 14% "large calendar
gaps" and collapsed reliability to 0.14 — every symbol on the exchange
refused for the crime of the market closing at 4pm. Separately, an overnight
return is many intra-session hourly sigmas wide and tripped the bad-print
detector.

Making the checks session-aware must not blind them: the defects they hunt
for are still real, and the tests below pin both directions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import DataConfig, UniverseConfig
from titan.data.quality import assess_quality
from titan.data.store import MarketDataStore

SESSION_HOURS = range(9, 16)  # 7 hourly bars per session
N_SESSIONS = 500


def _hourly_equity_frame(seed: int = 3) -> pd.DataFrame:
    """Hourly bars with genuine session structure: overnight gaps and jumps."""
    rng = np.random.default_rng(seed)
    stamps = [
        day + pd.Timedelta(hours=h)
        for day in pd.bdate_range("2024-01-02", periods=N_SESSIONS)
        for h in SESSION_HOURS
    ]
    idx = pd.DatetimeIndex(stamps, tz="UTC")
    n = len(idx)

    is_open = np.r_[True, idx.normalize()[1:] != idx.normalize()[:-1]]
    ret = rng.normal(0, 0.004, n)
    ret[is_open] = rng.normal(0, 0.012, is_open.sum())  # overnight moves are wider
    close = 100 * np.exp(np.cumsum(ret))

    return pd.DataFrame(
        {
            "open": close * 0.9995,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": rng.integers(1e5, 5e5, n).astype(float),
        },
        index=idx,
    )


def _daily_frame(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.DatetimeIndex(pd.bdate_range("2015-01-02", periods=2000), tz="UTC")
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, len(idx))))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(1e6, 5e6, len(idx)).astype(float),
        },
        index=idx,
    )


# ------------------------------------------------------------- the bug ----


def test_naive_qc_fails_hourly_equity_bars_on_their_session_boundaries():
    """The reported failure: reliability 0.14 from ~14% 'gaps' (1 of 7 bars)."""
    report = assess_quality("TEST", _hourly_equity_frame(), min_bars=3000)

    assert report.reliability < 0.2
    assert any("large calendar gaps" in issue for issue in report.issues)


def test_session_aware_qc_accepts_the_same_frame():
    report = assess_quality(
        "TEST", _hourly_equity_frame(), min_bars=3000, intraday_sessions=True
    )

    assert report.reliability > 0.95, report.issues
    assert report.checks["calendar_gaps"] == pytest.approx(1.0)
    assert report.checks["return_outliers"] == pytest.approx(1.0)


# ------------------------------------------- still catches real defects ----


def test_session_aware_qc_still_catches_a_hole_inside_a_session():
    """A hole within a session is missing data, not the market being closed.

    The check promises *large* gaps (>5x the median step), so this drops a
    contiguous block: a single absent bar is a 2x step and passes by design.
    """
    df = _hourly_equity_frame()
    # Remove 10:00-14:00 from a quarter of sessions, leaving a 6-hour hole.
    midday = df.index.hour.isin([10, 11, 12, 13, 14])
    drop = midday & (df.index.dayofyear % 4 == 0)
    holed = df[~drop]

    report = assess_quality("TEST", holed, min_bars=3000, intraday_sessions=True)

    assert report.checks["calendar_gaps"] < 1.0
    assert any("large calendar gaps" in issue for issue in report.issues)


def test_session_aware_qc_still_catches_a_bad_print_inside_a_session():
    df = _hourly_equity_frame()
    mid = df.index.hour == 13
    corrupt = df.copy()
    # A decimal-shift error on 1% of intra-session bars.
    corrupt.loc[mid, "close"] = corrupt.loc[mid, "close"] * 10

    report = assess_quality("TEST", corrupt, min_bars=3000, intraday_sessions=True)

    assert report.checks["return_outliers"] < 1.0
    assert any("robust sigmas" in issue for issue in report.issues)


def test_session_aware_qc_still_catches_a_bad_print_at_the_open():
    """Overnight returns are scored against their own population, not ignored."""
    df = _hourly_equity_frame()
    opens = df.index.hour == 9
    corrupt = df.copy()
    corrupt.loc[opens, "close"] = corrupt.loc[opens, "close"] * 10

    report = assess_quality("TEST", corrupt, min_bars=3000, intraday_sessions=True)

    assert report.checks["return_outliers"] < 1.0


# ----------------------------------------------------- daily is untouched --


def test_daily_frames_score_identically_either_way():
    """The flag must be inert on daily bars — no existing result may move."""
    df = _daily_frame()

    naive = assess_quality("TEST", df, min_bars=1500)
    aware = assess_quality("TEST", df, min_bars=1500, intraday_sessions=True)

    assert naive.reliability == pytest.approx(aware.reliability)
    assert naive.checks == pytest.approx(aware.checks)


# ---------------------------------------------------------- store wiring --


def _store(timeframe: str, asset_class: str) -> MarketDataStore:
    data = DataConfig(
        provider="synthetic",
        timeframe=timeframe,
        acknowledge_unvalidated_timeframe=True,  # 5m is refused without it
    )
    universe = UniverseConfig(
        name="t",
        benchmark="SPY",
        instruments=[
            {"symbol": "SPY", "asset_class": "etf", "sector": "broad"},
            {"symbol": "XXX", "asset_class": asset_class, "sector": "s"},
        ],
    )
    return MarketDataStore(data, universe, seed=7)


@pytest.mark.parametrize(
    ("timeframe", "asset_class", "expected"),
    [
        ("1h", "equity", True),    # market closes -> session-aware
        ("1h", "crypto", False),   # 24/7 -> no session boundaries to excuse
        ("1d", "equity", False),   # daily -> the naive checks are correct
        ("1wk", "equity", False),
        ("5m", "equity", True),
    ],
)
def test_store_resolves_the_session_flag(timeframe, asset_class, expected):
    assert _store(timeframe, asset_class)._intraday_sessions is expected
