"""End-to-end: the full research pipeline on a small synthetic universe.

This is the test that proves the platform hangs together: data -> features ->
labels -> purged walk-forward -> calibrated ensemble -> regime gating ->
signals -> risk -> portfolio simulation -> robustness statistics -> artifacts.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from titan.artifacts import write_scan_artifacts, write_walkforward_artifacts
from titan.backtest.costs import CostModel
from titan.backtest.walkforward import WalkForwardRunner
from titan.features.pipeline import FeatureMatrixBuilder
from titan.regime.detector import RegimeDetector
from titan.scanner.scanner import MarketScanner
from titan.signals.generator import SignalGenerator


@pytest.fixture(scope="module")
def wf_run(cfg, dataset):
    runner = WalkForwardRunner(cfg)
    panel = FeatureMatrixBuilder(cfg.features).build(dataset)
    report = runner.run(dataset, panel=panel)
    return runner, panel, report


@pytest.mark.slow
def test_walkforward_report_complete(wf_run):
    _, _, report = wf_run
    assert len(report.folds) == 2
    assert 0.3 < report.pooled_auc < 0.9
    assert 0 < report.pooled_brier < 0.5
    assert np.isfinite(report.backtest.summary.sharpe)
    assert report.backtest.equity.iloc[0] > 0
    assert (report.backtest.equity > 0).all()
    assert report.bootstrap.n_sims == 1000
    assert report.dsr_sensitivity["n_trials=1"] >= report.dsr_sensitivity["n_trials=25"]
    assert report.importance is not None and len(report.importance) > 0
    assert not report.decisions.empty
    # regime table covers the OOS window
    assert report.regimes.index.min() <= report.backtest.equity.index[0]


@pytest.mark.slow
def test_oos_probabilities_have_skill(wf_run):
    """Pooled OOS: gated bucket must beat base rate (the planted signal is real)."""
    _, _, report = wf_run
    d = report.decisions
    base = d["label"].mean()
    gated = d[d["p"] >= 0.55]
    assert len(gated) > 30
    assert gated["label"].mean() > base + 0.03


@pytest.mark.slow
def test_signals_fully_populated_and_consistent(wf_run):
    _, _, report = wf_run
    assert report.signals, "no signals emitted over the whole OOS window"
    for s in report.signals[:25]:
        assert s.probability >= s.threshold_used
        assert s.ev_after_costs > 0
        assert s.stop_loss < s.optimal_limit_entry
        assert s.position_size_fraction > 0
        assert s.market_regime.value != "crash"
        assert s.reasoning
        assert s.expected_holding_bars > 0
        d = s.to_dict()
        json.dumps(d)  # fully serializable
        assert set(d) >= {
            "confidence_score", "trade_grade", "entry_zone", "stop_loss", "atr_stop",
            "take_profit_levels", "position_size_fraction", "risk_percentage",
            "expected_holding_bars", "expected_volatility", "risk_reward",
            "mae_estimate", "mfe_estimate", "historical_similarity",
            "institutional_score", "market_regime", "supporting_evidence",
            "conflicting_evidence", "reasoning", "outcome_quantiles",
        }


@pytest.mark.slow
def test_trades_respect_portfolio_constraints(wf_run, cfg, dataset):
    _, _, report = wf_run
    trades = report.backtest.trades
    if not trades:
        pytest.skip("no trades this run")
    for t in trades:
        assert t.size_fraction <= cfg.risk.max_position_weight + 1e-9
        assert t.bars_held <= cfg.labels.horizon_bars + 1
    # exposure never exceeds the gross cap (small tolerance for close-vs-open marks)
    assert (report.backtest.exposure <= cfg.backtest.max_gross_exposure + 0.10).all()


@pytest.mark.slow
def test_artifacts_written_and_loadable(wf_run, cfg, dataset, tmp_path_factory):
    runner, panel, report = wf_run
    out = tmp_path_factory.mktemp("artifacts")
    write_walkforward_artifacts(out, cfg, dataset, report)

    for name in ("report.json", "signals.json", "trades.json", "equity.csv",
                 "regimes.csv", "manifest.json", "correlation.json", "universe.json"):
        assert (out / name).exists(), name
    loaded = json.loads((out / "report.json").read_text())
    assert loaded["pooled_auc"] == pytest.approx(report.pooled_auc, abs=1e-4)
    eq = pd.read_csv(out / "equity.csv")
    assert len(eq) == len(report.backtest.equity)

    # scanner on the last bar with last-fold artifacts
    a = runner.last_fold_artifacts
    generator = SignalGenerator(cfg.signals, cfg.labels, cfg.risk,
                                CostModel(cfg.backtest.costs), cfg.backtest.max_positions)
    detector = RegimeDetector(cfg.regime, seed=cfg.run.seed)
    detector.fit(dataset.benchmark_frame.iloc[:-63])
    scanner = MarketScanner(cfg, a["ensemble"], a["selected"], generator, detector,
                            explainer=a["explainer"], analogues=a["analogues"])
    scan = scanner.scan(dataset, panel)
    assert len(scan.rows) == len(dataset.frames)
    assert all(r.status for r in scan.rows)
    write_scan_artifacts(out, scan)
    scan_loaded = json.loads((out / "scan.json").read_text())
    assert scan_loaded["regime"]["regime"]


@pytest.mark.slow
def test_dashboard_api_serves_artifacts(wf_run, cfg, dataset, tmp_path_factory):
    from fastapi.testclient import TestClient

    from titan.server.app import create_app

    _runner, _panel, report = wf_run
    out = tmp_path_factory.mktemp("artifacts_api")
    write_walkforward_artifacts(out, cfg, dataset, report)

    client = TestClient(create_app(out))
    assert client.get("/health").status_code == 200
    page = client.get("/")
    assert page.status_code == 200 and "TITAN" in page.text
    for ep in ("/api/report", "/api/equity", "/api/regimes", "/api/signals",
               "/api/correlation", "/api/manifest"):
        r = client.get(ep)
        assert r.status_code == 200, ep
    assert client.get("/api/scan").status_code == 404  # not written in this tmp dir
