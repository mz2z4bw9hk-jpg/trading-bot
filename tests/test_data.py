"""Data layer: generator realism, schema normalization, QC scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.data.quality import assess_quality
from titan.data.schema import SchemaError, normalize_ohlcv
from titan.data.synthetic import STATE_PARAMS, SyntheticMarket


def test_synthetic_ohlc_internally_consistent(synthetic_result):
    for sym, f in synthetic_result.frames.items():
        assert (f["high"] >= f[["open", "close"]].max(axis=1) - 1e-9).all(), sym
        assert (f["low"] <= f[["open", "close"]].min(axis=1) + 1e-9).all(), sym
        assert (f[["open", "high", "low", "close"]] > 0).all().all(), sym


def test_synthetic_regimes_have_distinct_vol(synthetic_result):
    idx_ret = np.log(synthetic_result.index_frame["close"]).diff()
    regs = synthetic_result.true_regimes
    vols = {r: idx_ret[regs == r].std() for r in ("bull", "bear") if (regs == r).sum() > 30}
    if len(vols) == 2:
        assert vols["bear"] > vols["bull"] * 1.3


def test_synthetic_deterministic():
    a = SyntheticMarket(["X", "Y"], bars=120, seed=5).generate()
    b = SyntheticMarket(["X", "Y"], bars=120, seed=5).generate()
    pd.testing.assert_frame_equal(a.frames["X"], b.frames["X"])


def test_normalize_ohlcv_roundtrip(ohlcv):
    messy = ohlcv.copy()
    messy.columns = ["Open", "HIGH", "low", "Close", "Volume"]
    messy.index = messy.index.tz_localize(None)  # naive in, aware out
    out = normalize_ohlcv(messy)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing


def test_normalize_rejects_garbage():
    with pytest.raises(SchemaError):
        normalize_ohlcv(pd.DataFrame({"foo": [1, 2]}))


def test_quality_flags_corruption(ohlcv):
    clean = assess_quality("CLEAN", ohlcv, min_bars=300)
    assert clean.reliability > 0.9

    corrupt = ohlcv.copy()
    # stale block + broken OHLC + a fake 10x spike
    corrupt.iloc[100:220, corrupt.columns.get_loc("close")] = 50.0
    corrupt.iloc[300:320, corrupt.columns.get_loc("high")] = 0.5 * corrupt["low"].iloc[300:320]
    corrupt.iloc[400, corrupt.columns.get_loc("close")] *= 10
    bad = assess_quality("BAD", corrupt, min_bars=300)
    assert bad.reliability < clean.reliability - 0.1
    assert bad.issues


def test_state_params_sane():
    for name, (mu, sigma) in STATE_PARAMS.items():
        assert sigma > 0, name
        assert abs(mu) < 0.05, name
