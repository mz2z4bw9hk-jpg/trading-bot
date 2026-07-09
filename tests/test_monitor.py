"""Monitoring: drift alarms fire when they should, stay quiet when they shouldn't."""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.monitor.compare import compare_returns, promotion_gate
from titan.monitor.drift import (
    PredictionTracker,
    feature_drift_report,
    population_stability_index,
)


def test_psi_zero_for_identical():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(3000)
    assert population_stability_index(x, x) < 0.01


def test_psi_large_for_shifted():
    rng = np.random.default_rng(1)
    ref = rng.standard_normal(3000)
    shifted = rng.standard_normal(3000) + 1.5
    assert population_stability_index(ref, shifted) > 0.25


def test_feature_drift_report_flags_only_drifted():
    rng = np.random.default_rng(2)
    X_ref = pd.DataFrame({"stable": rng.standard_normal(2000),
                          "drifted": rng.standard_normal(2000)})
    X_live = pd.DataFrame({"stable": rng.standard_normal(500),
                           "drifted": rng.standard_normal(500) + 2.0})
    report = feature_drift_report(X_ref, X_live)
    assert "drifted" in report["alerts"]
    assert "stable" not in report["alerts"]


def test_tracker_calibration_and_cusum():
    tracker = PredictionTracker(baseline_brier=0.23)
    rng = np.random.default_rng(3)
    # well-calibrated phase: outcome ~ Bernoulli(p)
    for i in range(150):
        p = float(rng.uniform(0.4, 0.8))
        tracker.log_prediction(f"d{i}", "AAA", p)
        tracker.resolve(f"d{i}", "AAA", int(rng.random() < p))
    alarm_before, _ = tracker.cusum_alarm()
    assert not alarm_before  # a calibrated stream must not trip the alarm

    # broken phase: model says 0.8, world says 20%
    for i in range(150, 260):
        tracker.log_prediction(f"d{i}", "AAA", 0.8)
        tracker.resolve(f"d{i}", "AAA", int(rng.random() < 0.2))
    alarm_after, stat = tracker.cusum_alarm()
    assert alarm_after
    assert stat > 0
    assert tracker.rolling_brier(100) > 0.3
    assert tracker.calibration_table()  # non-empty


def test_compare_returns_detects_better_strategy():
    idx = pd.bdate_range("2020-01-01", periods=800, tz="UTC")
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.01, 800)
    prod = pd.Series(base, index=idx)
    chall = pd.Series(base + 0.0012, index=idx)  # strictly better every day
    out = compare_returns(chall, prod, n_sims=500, seed=5)
    assert out["p_value"] < 0.05
    assert out["challenger_sharpe"] > out["production_sharpe"]


def test_compare_returns_no_false_positive():
    idx = pd.bdate_range("2020-01-01", periods=800, tz="UTC")
    rng = np.random.default_rng(6)
    prod = pd.Series(rng.normal(0.0003, 0.01, 800), index=idx)
    chall = prod + rng.normal(0, 0.0001, 800)  # same strategy + dust
    out = compare_returns(chall, prod, n_sims=500, seed=7)
    assert out["p_value"] > 0.05


def test_promotion_gate_requires_everything():
    good_cmp = {"p_value": 0.01, "challenger_sharpe": 1.2, "production_sharpe": 0.8}
    ok, reasons = promotion_gate(good_cmp, {"max_drawdown": -0.10}, {"max_drawdown": -0.12})
    assert ok and not reasons

    bad_p = {**good_cmp, "p_value": 0.30}
    ok, reasons = promotion_gate(bad_p, {"max_drawdown": -0.10}, {"max_drawdown": -0.12})
    assert not ok and any("significant" in r for r in reasons)

    worse_dd = promotion_gate(good_cmp, {"max_drawdown": -0.30}, {"max_drawdown": -0.10})
    assert not worse_dd[0]

    worse_sharpe = promotion_gate(
        {**good_cmp, "challenger_sharpe": 0.5}, {"max_drawdown": -0.1}, {"max_drawdown": -0.1}
    )
    assert not worse_sharpe[0]


def test_insufficient_overlap():
    idx = pd.bdate_range("2020-01-01", periods=10, tz="UTC")
    a = pd.Series(np.zeros(10), index=idx)
    out = compare_returns(a, a)
    assert out["verdict"] == "insufficient_overlap"
    ok, _reasons = promotion_gate(out, {}, {})
    assert not ok
