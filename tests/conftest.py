"""Shared fixtures: small synthetic markets sized for fast, deterministic tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import TitanConfig
from titan.core.types import AssetClass, Instrument, Universe
from titan.data.store import MarketDataset
from titan.data.synthetic import SyntheticMarket


def make_config(**overrides) -> TitanConfig:
    base = {
        "run": {"seed": 11},
        "data": {"provider": "synthetic", "bars": 900, "min_history_bars": 300},
        "universe": {
            "benchmark": "INDEX",
            "instruments": [
                {"symbol": s, "sector": sec}
                for s, sec in [
                    ("AAA", "tech"), ("BBB", "tech"), ("CCC", "fin"),
                    ("DDD", "fin"), ("EEE", "energy"), ("FFF", "energy"),
                ]
            ],
        },
        "features": {
            "momentum_windows": [5, 21, 63],
            "trend_windows": [10, 21],
            "meanrev_windows": [5, 21],
            "vol_windows": [5, 21, 63],
            "volume_windows": [21],
            "structure_windows": [21, 63],
            "cross_windows": [21, 63],
            "max_features": 24,
        },
        "labels": {"horizon_bars": 10, "vol_span": 21},
        "cv": {"n_folds": 2, "min_train_bars": 320, "test_bars": 120, "embargo_bars": 5},
        "model": {"members": ["hgb", "logistic"], "tuning_iterations": 0},
        "regime": {"min_train_bars": 200},
    }

    def merge(a: dict, b: dict) -> dict:
        out = dict(a)
        for k, v in b.items():
            out[k] = merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
        return out

    return TitanConfig.model_validate(merge(base, overrides))


@pytest.fixture(scope="session")
def cfg() -> TitanConfig:
    return make_config()


@pytest.fixture(scope="session")
def synthetic_result(cfg):
    market = SyntheticMarket(
        symbols=[i.symbol for i in cfg.universe.instruments],
        bars=cfg.data.bars,
        seed=cfg.run.seed,
        drift_sigma=0.0009,  # stronger planted signal: tests assert *detection*,
    )                        # and must not be knife-edge against seed jitter
    return market.generate()


@pytest.fixture(scope="session")
def dataset(cfg, synthetic_result) -> MarketDataset:
    universe = Universe(
        instruments=[
            Instrument(symbol=i.symbol, asset_class=AssetClass.EQUITY, sector=i.sector)
            for i in cfg.universe.instruments
        ],
        benchmark=cfg.universe.benchmark,
    )
    reliability = dict.fromkeys(synthetic_result.frames, 1.0)
    return MarketDataset(
        frames=synthetic_result.frames,
        benchmark_frame=synthetic_result.index_frame,
        universe=universe,
        reliability=reliability,
        true_regimes=synthetic_result.true_regimes,
    )


@pytest.fixture(scope="session")
def ohlcv(synthetic_result) -> pd.DataFrame:
    return synthetic_result.frames["AAA"]


def make_trend_frame(n: int = 300, daily: float = 0.002, seed: int = 3) -> pd.DataFrame:
    """Deterministic-ish trending frame for hand-checkable tests."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    ret = daily + 0.005 * rng.standard_normal(n)
    close = 100 * np.exp(np.cumsum(ret))
    prev = np.concatenate([[100.0], close[:-1]])
    high = np.maximum(prev, close) * 1.004
    low = np.minimum(prev, close) * 0.996
    return pd.DataFrame(
        {"open": prev, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1e6, 2e6, n)},
        index=idx,
    )
