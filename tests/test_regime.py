"""Regime detection: real skill against known ground truth, causally.

The separation test uses a dedicated long benchmark (1800 bars) so the fit
window genuinely contains bear episodes — a detector cannot learn states it
has never seen, and neither could a production deployment. That data-depth
requirement is a documented operating constraint, not a test convenience.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.data.synthetic import SyntheticMarket
from titan.regime.detector import RegimeDetector

BULLISH = {"strong_bull", "bull", "weak_bull", "accumulation"}
BEARISH = {"bear", "crash", "distribution", "correction"}


def _mapped(reg_series: pd.Series) -> pd.Series:
    return reg_series.map(
        lambda r: "bull" if r in BULLISH else ("bear" if r in BEARISH else "range")
    )


@pytest.fixture(scope="module")
def long_market():
    return SyntheticMarket(symbols=["MKT"], bars=1800, seed=21).generate()


def test_detector_separates_bull_from_bear(cfg, long_market):
    bench = long_market.index_frame
    det = RegimeDetector(cfg.regime, seed=cfg.run.seed).fit(bench.iloc[:1000])
    table = det.transform(bench)
    truth = long_market.true_regimes.reindex(table.index).map(
        {"bull": "bull", "recovery": "bull", "bear": "bear", "crash": "bear", "range": "range"}
    )
    bearish_post = table["p_bear"] + table["p_turbulent"]
    separation = bearish_post[truth == "bear"].mean() - bearish_post[truth == "bull"].mean()
    assert separation > 0.15, f"bear/bull posterior separation too weak: {separation:.3f}"

    mapped = _mapped(table["regime"])
    bear_recall = (mapped[truth == "bear"] == "bear").mean()
    bear_base = (truth == "bear").mean()
    bear_precision = (truth[mapped == "bear"] == "bear").mean()
    assert bear_recall > 0.45
    assert bear_precision > bear_base  # better than guessing


def test_detector_is_causal(cfg, dataset):
    """Frozen detector's classification at t must not change when future data is removed."""
    bench = dataset.benchmark_frame
    det = RegimeDetector(cfg.regime, seed=cfg.run.seed).fit(bench.iloc[:500])
    full = det.transform(bench.iloc[:800])
    cut = det.transform(bench.iloc[:700])
    ts = bench.index[699]
    assert full.loc[ts, "regime"] == cut.loc[ts, "regime"]
    for col in ("p_bull", "p_bear", "p_range", "p_turbulent"):
        assert np.isclose(float(full.loc[ts, col]), float(cut.loc[ts, col]), atol=1e-9)


def test_vol_state_extreme_during_crash(cfg, dataset):
    bench = dataset.benchmark_frame
    det = RegimeDetector(cfg.regime, seed=cfg.run.seed).fit(bench.iloc[:500])
    table = det.transform(bench)
    crash_days = dataset.true_regimes.reindex(table.index) == "crash"
    if crash_days.sum() >= 5:
        elevated = table.loc[crash_days, "vol_state"].isin(
            ["high_volatility", "extreme_volatility"]
        )
        assert elevated.mean() > 0.5


def test_transform_requires_fit(cfg, dataset):
    det = RegimeDetector(cfg.regime)
    import pytest

    with pytest.raises(RuntimeError):
        det.transform(dataset.benchmark_frame)


def test_snapshot_shape(cfg, dataset):
    det = RegimeDetector(cfg.regime, seed=cfg.run.seed).fit(dataset.benchmark_frame.iloc[:500])
    snap = det.snapshot(dataset.benchmark_frame)
    assert 0.0 <= snap.confidence <= 1.0
    assert abs(sum(snap.posteriors.values()) - 1.0) < 1e-6
