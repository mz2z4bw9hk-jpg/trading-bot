"""Model registry lifecycle and dataset store."""

from __future__ import annotations

import pytest

from titan.core.config import DataConfig, UniverseConfig, UniverseItem
from titan.data.store import MarketDataStore
from titan.models.registry import (
    STATUS_CANDIDATE,
    STATUS_PRODUCTION,
    STATUS_RETIRED,
    ModelRegistry,
)


def test_registry_lifecycle(tmp_path):
    reg = ModelRegistry(tmp_path / "store")
    v1 = reg.save({"model": "a"}, metrics={"sharpe": 0.5}, description="first")
    v2 = reg.save({"model": "b"}, metrics={"sharpe": 0.9}, description="second")
    assert (v1, v2) == ("v001", "v002")
    assert reg.production_version() is None

    reg.promote(v1)
    assert reg.production_version() == v1
    model, manifest = reg.load(None)
    assert model == {"model": "a"}
    assert manifest.status == STATUS_PRODUCTION

    reg.promote(v2)
    assert reg.production_version() == v2
    history = {m.version: m.status for m in reg.history()}
    assert history[v1] == STATUS_RETIRED
    assert history[v2] == STATUS_PRODUCTION

    with pytest.raises(LookupError):
        reg.load("v999")


def test_registry_candidate_status(tmp_path):
    reg = ModelRegistry(tmp_path / "store")
    v = reg.save({"m": 1})
    _, manifest = reg.load(v)
    assert manifest.status == STATUS_CANDIDATE


def test_store_loads_universe_and_excludes_missing(tmp_path):
    data_cfg = DataConfig(provider="synthetic", bars=600, min_history_bars=300,
                          cache_dir=tmp_path / "cache")
    uni = UniverseConfig(
        benchmark="INDEX",
        instruments=[UniverseItem(symbol="AAA"), UniverseItem(symbol="BBB")],
    )
    store = MarketDataStore(data_cfg, uni, seed=3)
    ds = store.load(use_cache=True)
    assert set(ds.symbols) == {"AAA", "BBB"}
    assert all(0 <= v <= 1 for v in ds.reliability.values())
    assert len(ds.benchmark_frame) == 600
    assert ds.true_regimes is not None

    # cache round-trip: second load hits parquet and matches
    ds2 = store.load(use_cache=True)
    assert ds2.frames["AAA"].equals(ds.frames["AAA"])


def test_cache_key_differentiates_generation_params(tmp_path):
    """Regression: a cached weak-signal market must never be served for a
    strong-signal config (cache key ignores nothing that shapes the data)."""
    uni = UniverseConfig(benchmark="INDEX", instruments=[UniverseItem(symbol="AAA")])
    weak_cfg = DataConfig(provider="synthetic", bars=400, min_history_bars=200,
                          cache_dir=tmp_path / "cache")
    strong_cfg = DataConfig(provider="synthetic", bars=400, min_history_bars=200,
                            cache_dir=tmp_path / "cache", synthetic_drift_sigma=0.002)
    weak = MarketDataStore(weak_cfg, uni, seed=3).load(use_cache=True)
    strong = MarketDataStore(strong_cfg, uni, seed=3).load(use_cache=True)
    assert not weak.frames["AAA"]["close"].equals(strong.frames["AAA"]["close"])
