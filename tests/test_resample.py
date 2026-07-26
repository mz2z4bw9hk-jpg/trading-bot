"""Aggregating bars: correctness, causality, and the volume repair.

Yahoo's hourly crypto series carries no volume on roughly half its bars, which
fails QC outright. Summing source bars into a coarser target bar recovers a
complete volume column — a target bar is empty only where every source bar in
it was. That is a real repair, not a relaxed check, so it has to be correct:
right aggregation, no bar spanning a market closure, and no timestamp that
would let a bar's own future leak into a feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import DataConfig, UniverseConfig
from titan.data.quality import assess_quality
from titan.data.resample import resample_ohlcv, zero_volume_fraction
from titan.data.store import MarketDataStore


def _hourly(n_days: int = 40, *, continuous: bool = True, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    if continuous:
        idx = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=n_days * 24, freq="h", tz="UTC"))
    else:
        idx = pd.DatetimeIndex(
            [
                day + pd.Timedelta(hours=h)
                for day in pd.bdate_range("2024-01-01", periods=n_days)
                for h in range(9, 16)
            ],
            tz="UTC",
        )
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.003, len(idx))))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.004,
            "low": close * 0.996,
            "close": close,
            "volume": rng.integers(1e4, 9e4, len(idx)).astype(float),
        },
        index=idx,
    )


# ------------------------------------------------------- aggregation ------


def test_ohlcv_aggregates_with_the_right_reducer():
    df = _hourly(n_days=2)
    out = resample_ohlcv(df, "4h", within_sessions=False)

    first = df.iloc[:4]
    assert out["open"].iloc[0] == pytest.approx(first["open"].iloc[0])
    assert out["high"].iloc[0] == pytest.approx(first["high"].max())
    assert out["low"].iloc[0] == pytest.approx(first["low"].min())
    assert out["close"].iloc[0] == pytest.approx(first["close"].iloc[-1])
    assert out["volume"].iloc[0] == pytest.approx(first["volume"].sum())


def test_aggregation_conserves_total_volume():
    df = _hourly()
    out = resample_ohlcv(df, "4h", within_sessions=False)
    assert out["volume"].sum() == pytest.approx(df["volume"].sum())


def test_bars_are_stamped_at_the_start_of_the_interval_they_cover():
    """Stamping right would date a bar before data inside it: look-ahead."""
    df = _hourly(n_days=2)
    out = resample_ohlcv(df, "4h", within_sessions=False)

    assert out.index[0] == df.index[0]
    # The bar stamped t must close with the last source bar strictly inside
    # [t, t+4h) — never with one at or beyond t+4h.
    window = df.loc[out.index[0] : out.index[0] + pd.Timedelta(hours=4)].iloc[:-1]
    assert out["close"].iloc[0] == pytest.approx(window["close"].iloc[-1])


def test_session_markets_never_fuse_bars_across_a_closure():
    """A bar spanning the overnight gap never traded as one bar."""
    df = _hourly(n_days=10, continuous=False)
    out = resample_ohlcv(df, "4h", within_sessions=True)

    # Every output bar's source rows must share one calendar date.
    assert len(out) == len(df.index.normalize().unique()) * 2  # 7h session -> 2 buckets
    assert out.index.hour.isin([8, 12]).all()


def test_continuous_markets_aggregate_straight_through():
    df = _hourly(n_days=10, continuous=True)
    out = resample_ohlcv(df, "4h", within_sessions=False)
    assert len(out) == pytest.approx(10 * 6, abs=1)


def test_empty_frame_survives():
    empty = _hourly(n_days=2).iloc[:0]
    assert resample_ohlcv(empty, "4h", within_sessions=False).empty


# ------------------------------------------------------ the volume repair --


def _with_alternating_zero_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's hourly crypto shape: volume on only half the bars."""
    out = df.copy()
    out.iloc[::2, out.columns.get_loc("volume")] = 0.0
    return out


def test_aggregation_repairs_a_half_empty_volume_column():
    sparse = _with_alternating_zero_volume(_hourly())
    assert zero_volume_fraction(sparse) == pytest.approx(0.5, abs=0.01)

    out = resample_ohlcv(sparse, "2h", within_sessions=False)

    assert zero_volume_fraction(out) == 0.0
    assert out["volume"].sum() == pytest.approx(sparse["volume"].sum())


def test_the_repair_is_what_makes_qc_pass():
    sparse = _with_alternating_zero_volume(_hourly(n_days=200))

    before = assess_quality("X", sparse, min_bars=1000)
    after = assess_quality("X", resample_ohlcv(sparse, "2h", within_sessions=False), min_bars=1000)

    assert before.reliability < 0.5
    assert after.reliability > 0.9, after.issues


def test_aggregation_does_not_invent_volume_where_there_was_none():
    """A target bar stays empty when every source bar in it was empty."""
    df = _hourly(n_days=10)
    blanked = df.copy()
    blanked["volume"] = 0.0

    out = resample_ohlcv(blanked, "4h", within_sessions=False)

    assert zero_volume_fraction(out) == 1.0


# ------------------------------------------------------------- config -----


def test_resample_source_must_be_finer_than_the_target():
    with pytest.raises(ValueError, match="must be a FINER interval"):
        DataConfig(timeframe="1h", resample_from="4h")


def test_resample_source_equal_to_target_is_rejected():
    with pytest.raises(ValueError, match="must be a FINER interval"):
        DataConfig(timeframe="1h", resample_from="1h")


def test_multi_hour_timeframes_are_reachable_only_by_aggregation():
    from titan.data.providers import YahooProvider

    with pytest.raises(ValueError, match="resample_from"):
        YahooProvider("2h")


# -------------------------------------------------------- store wiring ----


def _store(timeframe: str, resample_from: str | None) -> MarketDataStore:
    data = DataConfig(provider="synthetic", timeframe=timeframe, resample_from=resample_from)
    universe = UniverseConfig(
        name="t",
        benchmark="BTC-USD",
        instruments=[{"symbol": "BTC-USD", "asset_class": "crypto", "sector": "crypto"}],
    )
    return MarketDataStore(data, universe, seed=7)


def test_store_requests_enough_source_bars_to_fill_the_target():
    """Asking for N target bars must fetch N x (source/target) source bars."""
    plain = _store("1h", None)
    resampled = _store("2h", "1h")

    assert resampled._source_bars == pytest.approx(plain._cfg.bars * 2, rel=0.01)


def test_store_leaves_bars_alone_without_resampling():
    store = _store("1h", None)
    assert store._source_bars == store._cfg.bars


# --------------------------------------------------------- cache identity --


def _cache_dir(timeframe: str, resample_from: str | None) -> str:
    return str(_store(timeframe, resample_from)._cache_dir)


def test_cache_is_keyed_by_timeframe():
    """Same symbol and bar count at a different interval is different data.

    The store caches post-resample frames. A key that ignores the interval
    hands 1d bars to a 1h run — identical shape, wrong market, and no
    downstream check can detect it.
    """
    assert _cache_dir("1d", None) != _cache_dir("1h", None)


def test_cache_is_keyed_by_the_resample_source():
    """4h aggregated from 1h is not the same frame as native 4h."""
    assert _cache_dir("4h", "1h") != _cache_dir("4h", None)


def test_cache_key_is_stable_for_one_configuration():
    assert _cache_dir("2h", "1h") == _cache_dir("2h", "1h")
