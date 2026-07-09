"""Config-mismatch guard: a model must never be scanned against a world it
never saw. (Discovered the hard way: bare `titan scan` defaults to
configs/default.yaml, silently mis-scanning a model validated elsewhere.)"""

from __future__ import annotations

import contextlib

import joblib

from titan.cli import main
from titan.core.config import load_config, research_fingerprint
from titan.models.registry import ModelRegistry


def _cfg_yaml(tmp_path, drift: float, name: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        f"""
run: {{ artifacts_dir: {tmp_path / "artifacts"}, log_level: WARNING }}
data: {{ provider: synthetic, cache_dir: {tmp_path / "cache"}, bars: 700,
         synthetic_drift_sigma: {drift} }}
model: {{ store_dir: {tmp_path / "models"} }}
universe:
  benchmark: IDX
  instruments:
    - {{ symbol: AAA, asset_class: equity, sector: tech }}
"""
    )
    return path


def test_fingerprint_tracks_the_world_not_the_search(tmp_path):
    a = load_config(_cfg_yaml(tmp_path, 0.004, "a"), {})
    same = load_config(_cfg_yaml(tmp_path, 0.004, "same"), {})
    other_market = load_config(_cfg_yaml(tmp_path, 0.010, "b"), {})

    assert research_fingerprint(a) == research_fingerprint(same)
    assert research_fingerprint(a) != research_fingerprint(other_market)
    # synthetic: the seed IS the market
    assert research_fingerprint(a) != research_fingerprint(
        load_config(_cfg_yaml(tmp_path, 0.004, "c"), {"run": {"seed": 13}})
    )
    # search intensity / gates / paths are NOT the world
    assert research_fingerprint(a) == research_fingerprint(
        load_config(_cfg_yaml(tmp_path, 0.004, "d"),
                    {"model": {"tuning_iterations": 99},
                     "signals": {"min_probability": 0.7},
                     "run": {"artifacts_dir": str(tmp_path / "elsewhere")}})
    )
    # cache location is irrelevant; label geometry is not
    assert research_fingerprint(a) == research_fingerprint(
        load_config(_cfg_yaml(tmp_path, 0.004, "e"), {"data": {"cache_dir": str(tmp_path / "x")}})
    )
    assert research_fingerprint(a) != research_fingerprint(
        load_config(_cfg_yaml(tmp_path, 0.004, "f"), {"labels": {"tp_sigma": 3.0}})
    )


def test_scan_refuses_config_mismatch(tmp_path, capsys):
    cfg_a = _cfg_yaml(tmp_path, 0.004, "world_a")
    cfg_b = _cfg_yaml(tmp_path, 0.010, "world_b")
    registry = ModelRegistry(tmp_path / "models")
    version = registry.save(
        {"stub": True},
        description="stub bundle for guard test",
        config_fingerprint=research_fingerprint(load_config(cfg_a, {})),
    )
    registry.promote(version)

    assert main(["scan", "--config", str(cfg_b)]) == 2
    err = capsys.readouterr().err
    assert "CONFIG MISMATCH" in err and "--allow-config-mismatch" in err

    # resolve path is guarded the same way
    assert main(["track", "resolve", "--config", str(cfg_b)]) == 2
    assert "CONFIG MISMATCH" in capsys.readouterr().err
    # status never runs inference, so it must not be blocked
    assert main(["track", "status", "--config", str(cfg_b)]) == 0


def test_legacy_manifests_without_fingerprint_pass(tmp_path, capsys):
    cfg_b = _cfg_yaml(tmp_path, 0.010, "world_b")
    registry = ModelRegistry(tmp_path / "models")
    version = registry.save({"stub": True}, description="pre-guard bundle")
    registry.promote(version)
    # no fingerprint recorded -> nothing to compare -> proceeds past the guard
    # (and then fails on the stub bundle, which is fine: we only assert the
    # guard itself didn't block)
    with contextlib.suppress(Exception):
        main(["scan", "--config", str(cfg_b)])
    assert "CONFIG MISMATCH" not in capsys.readouterr().err


def test_override_flag_bypasses_guard(tmp_path, capsys):
    cfg_a = _cfg_yaml(tmp_path, 0.004, "world_a")
    cfg_b = _cfg_yaml(tmp_path, 0.010, "world_b")
    registry = ModelRegistry(tmp_path / "models")
    version = registry.save(
        {"stub": True},
        config_fingerprint=research_fingerprint(load_config(cfg_a, {})),
    )
    registry.promote(version)
    # stub bundle cannot actually scan; the guard letting it through is the test
    with contextlib.suppress(Exception):
        main(["scan", "--config", str(cfg_b), "--allow-config-mismatch"])
    assert "CONFIG MISMATCH" not in capsys.readouterr().err


def test_registry_roundtrips_fingerprint(tmp_path):
    registry = ModelRegistry(tmp_path / "models")
    version = registry.save(joblib.__version__, config_fingerprint="abc123def456")
    _, manifest = registry.load(version)
    assert manifest.config_fingerprint == "abc123def456"
