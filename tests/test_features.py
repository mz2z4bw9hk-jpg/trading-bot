"""Feature engineering: math spot-checks, pruning, importance, panel shape."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_trend_frame
from titan.features.pipeline import FeatureMatrixBuilder, build_default_registry
from titan.features.rolling import (
    atr,
    efficiency_ratio,
    rolling_slope_stats,
    wilder_rsi,
)
from titan.features.selection import redundancy_prune, univariate_ic


def test_rolling_slope_tstat_detects_trend():
    up = make_trend_frame(daily=0.004)
    flat = make_trend_frame(daily=0.0)
    _, t_up = rolling_slope_stats(np.log(up["close"]), 21)
    _, t_flat = rolling_slope_stats(np.log(flat["close"]), 21)
    assert t_up.iloc[-1] > 2.0
    assert abs(t_flat.tail(50).mean()) < abs(t_up.tail(50).mean())


def test_rolling_slope_matches_polyfit():
    rng = np.random.default_rng(0)
    y = pd.Series(np.cumsum(rng.standard_normal(80)))
    slope, _ = rolling_slope_stats(y, 20)
    expected = np.polyfit(np.arange(20), y.iloc[-20:].to_numpy(), 1)[0]
    assert slope.iloc[-1] == pytest.approx(expected, rel=1e-9)


def test_efficiency_ratio_bounds():
    frame = make_trend_frame()
    er = efficiency_ratio(frame["close"], 10).dropna()
    assert ((er >= 0) & (er <= 1 + 1e-9)).all()
    # a perfectly straight line has ER = 1
    line = pd.Series(np.arange(1, 60, dtype=float))
    assert efficiency_ratio(line, 10).iloc[-1] == pytest.approx(1.0)


def test_rsi_extremes():
    up = pd.Series(np.linspace(100, 200, 60))
    down = pd.Series(np.linspace(200, 100, 60))
    assert wilder_rsi(up, 14).iloc[-1] > 95
    assert wilder_rsi(down, 14).iloc[-1] < 5


def test_atr_positive_and_scaled(ohlcv):
    a = atr(ohlcv, 14).dropna()
    assert (a > 0).all()
    assert (a < ohlcv["close"].reindex(a.index) * 0.2).all()


def test_panel_shape_and_families(cfg, dataset):
    builder = FeatureMatrixBuilder(cfg.features)
    panel = builder.build(dataset)
    assert panel.X.index.names == ["date", "symbol"]
    n_symbols = len(dataset.frames)
    assert panel.X.index.get_level_values(1).nunique() == n_symbols
    fams = set(panel.families.values())
    for expected in ("momentum", "trend", "meanrev", "volatility", "liquidity",
                     "structure", "cross_sectional"):
        assert expected in fams
    # warm-up rows dropped: coverage floor holds
    assert (panel.X.notna().mean(axis=1) >= 0.85).all()


def test_redundancy_prune_drops_duplicates():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(2000)
    X = pd.DataFrame({
        "orig": base,
        "clone": base * 1.0000001 + 1e-9 * rng.standard_normal(2000),
        "indep": rng.standard_normal(2000),
    })
    priority = pd.Series({"orig": 2.0, "clone": 1.0, "indep": 0.5})
    result = redundancy_prune(X, threshold=0.95, priority=priority)
    assert "orig" in result.kept
    assert "indep" in result.kept
    assert result.dropped.get("clone") == "orig"


def test_univariate_ic_finds_informative():
    rng = np.random.default_rng(2)
    x_good = rng.standard_normal(3000)
    y = pd.Series((x_good + rng.standard_normal(3000) > 0).astype(int))
    X = pd.DataFrame({"good": x_good, "noise": rng.standard_normal(3000)})
    ic = univariate_ic(X, y)
    assert abs(ic["good"]) > 0.3
    assert abs(ic["noise"]) < 0.08


def test_registry_names_unique(cfg):
    registry = build_default_registry(cfg.features)
    assert len(registry.names) == len(set(registry.names))
    assert len(registry) > 30
