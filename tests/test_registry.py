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


def test_production_is_scoped_per_research_world(tmp_path):
    """One champion per config fingerprint, not one per registry."""
    reg = ModelRegistry(tmp_path / "store")
    a1 = reg.save({"m": "a1"}, config_fingerprint="world_a")
    b1 = reg.save({"m": "b1"}, config_fingerprint="world_b")
    reg.promote(a1)
    reg.promote(b1)

    assert reg.production_version("world_a") == a1
    assert reg.production_version("world_b") == b1
    # neither promotion retired the other world's champion
    assert {m.version: m.status for m in reg.history()}[a1] == STATUS_PRODUCTION

    a2 = reg.save({"m": "a2"}, config_fingerprint="world_a")
    reg.promote(a2)
    history = {m.version: m.status for m in reg.history()}
    assert history[a1] == STATUS_RETIRED      # same world: replaced
    assert history[b1] == STATUS_PRODUCTION   # other world: untouched
    assert reg.production_version("world_a") == a2


def test_a_foreign_champion_does_not_block_a_new_world(tmp_path):
    """The deadlock this scoping exists to break.

    `validate` promotes outright when its config has no incumbent, and runs the
    champion/challenger bootstrap only when it does. Globally there IS a
    production model here; for world B there is not, so B's first model must be
    promoted rather than bootstrapped against a return series from another
    market — a comparison with no shared bars, which the gate can only reject.
    """
    reg = ModelRegistry(tmp_path / "store")
    reg.promote(reg.save({"m": "a"}, config_fingerprint="world_a"))

    assert reg.production_version() is not None
    assert reg.production_version("world_b") is None


def test_load_resolves_production_within_a_world(tmp_path):
    reg = ModelRegistry(tmp_path / "store")
    reg.promote(reg.save({"m": "a"}, config_fingerprint="world_a"))
    reg.promote(reg.save({"m": "b"}, config_fingerprint="world_b"))

    assert reg.load(None, fingerprint="world_a")[0] == {"m": "a"}
    assert reg.load(None, fingerprint="world_b")[0] == {"m": "b"}
    # an explicit version wins over the fingerprint: the caller already decided
    assert reg.load("v001", fingerprint="world_b")[0] == {"m": "a"}
    with pytest.raises(LookupError, match="no production model for research config"):
        reg.load(None, fingerprint="world_c")


def test_legacy_manifests_are_usable_by_any_world(tmp_path):
    """A pre-fingerprint model records nothing, and nothing is not a mismatch."""
    reg = ModelRegistry(tmp_path / "store")
    legacy = reg.save({"m": "old"})
    reg.promote(legacy)
    assert reg.production_version("any_world") == legacy
    assert reg.versions_for_fingerprint("any_world") == [legacy]


def test_versions_for_fingerprint_lists_candidates_but_not_retired(tmp_path):
    reg = ModelRegistry(tmp_path / "store")
    old = reg.save({"m": 1}, config_fingerprint="world_a")
    reg.promote(old)
    new = reg.save({"m": 2}, config_fingerprint="world_a")
    reg.promote(new)                                    # retires `old`
    cand = reg.save({"m": 3}, config_fingerprint="world_a")
    reg.save({"m": 4}, config_fingerprint="world_b")     # different world

    assert reg.versions_for_fingerprint("world_a") == [new, cand]


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
