"""Signal generation: the gates must actually gate."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.backtest.costs import CostModel
from titan.core.config import CostConfig, LabelConfig, RiskConfig, SignalConfig
from titan.core.types import Regime, TradeGrade, VolState
from titan.signals.analogues import AnalogueIndex, AnalogueReport
from titan.signals.generator import SignalGenerator


@pytest.fixture(scope="module")
def gen() -> SignalGenerator:
    return SignalGenerator(
        SignalConfig(), LabelConfig(), RiskConfig(), CostModel(CostConfig()), max_positions=8
    )


@pytest.fixture(scope="module")
def frame(ohlcv) -> pd.DataFrame:
    return ohlcv.iloc[:400]


def _analogue(hit=0.7) -> AnalogueReport:
    return AnalogueReport(
        similarity=0.8, n=50, hit_rate=hit,
        ret_quantiles={"p10": -0.03, "p25": -0.015, "p50": 0.01, "p75": 0.03, "p90": 0.05},
        median_bars_held=6.0, mae_p75=-0.02, mfe_p50=0.025,
    )


def _kwargs(frame, **over):
    base = {
        "symbol": "AAA", "date": frame.index[-1], "probability": 0.70, "uncertainty": 0.05,
        "frame": frame, "feature_row": pd.Series(dtype=float), "regime": Regime.BULL,
        "vol_state": VolState.NORMAL, "regime_confidence": 0.8, "analogue": _analogue(),
        "reliability": 1.0, "explainer": None, "model_version": "test",
    }
    base.update(over)
    return base


def test_high_probability_signal_emitted(gen, frame):
    sig = gen.generate(**_kwargs(frame))
    assert sig is not None
    assert sig.trade_grade in {TradeGrade.A_PLUS, TradeGrade.A, TradeGrade.B_PLUS, TradeGrade.B}
    assert sig.ev_after_costs > 0
    assert sig.stop_loss < sig.market_entry < sig.take_profit_levels[0]
    assert sig.take_profit_levels == sorted(sig.take_profit_levels)
    assert 0 < sig.position_size_fraction <= 0.15
    assert sig.risk_percentage > 0
    assert sig.reasoning and sig.outcome_quantiles


def test_below_threshold_blocked(gen, frame):
    assert gen.generate(**_kwargs(frame, probability=0.52)) is None


def test_crash_regime_blocked(gen, frame):
    assert gen.generate(**_kwargs(frame, regime=Regime.CRASH, probability=0.9)) is None


def test_uncertainty_gate(gen, frame):
    assert gen.generate(**_kwargs(frame, uncertainty=0.5)) is None


def test_hostile_regime_requires_more_edge(gen):
    # Low sigma keeps the derived threshold above the probability floor, so
    # the regime tightening is visible rather than clipped by the floor.
    sigma = 0.003
    tau_bull, _ = gen.adaptive_threshold(sigma, Regime.BULL)
    tau_bear, _ = gen.adaptive_threshold(sigma, Regime.BEAR)
    assert tau_bull > gen._cfg.min_probability  # formula, not floor, is binding
    assert tau_bear > tau_bull


def test_threshold_rises_with_costs(frame):
    cheap = SignalGenerator(SignalConfig(), LabelConfig(), RiskConfig(),
                            CostModel(CostConfig(commission_bps=0, spread_bps=0)), 8)
    expensive = SignalGenerator(SignalConfig(), LabelConfig(), RiskConfig(),
                                CostModel(CostConfig(commission_bps=20, spread_bps=20)), 8)
    sigma = 0.01
    assert expensive.adaptive_threshold(sigma, Regime.BULL)[0] > \
           cheap.adaptive_threshold(sigma, Regime.BULL)[0]


def test_ev_gate_blocks_negative_expectancy(frame):
    """With brutal costs, even p=0.62 has negative EV -> no signal."""
    gen = SignalGenerator(
        SignalConfig(min_probability=0.51), LabelConfig(), RiskConfig(),
        CostModel(CostConfig(commission_bps=100, spread_bps=100)), 8,
    )
    sig = gen.generate(**_kwargs(frame, probability=0.62))
    assert sig is None


def test_grade_monotone_in_probability(gen, frame):
    lo = gen.generate(**_kwargs(frame, probability=0.62))
    hi = gen.generate(**_kwargs(frame, probability=0.95))
    assert hi is not None
    if lo is not None:
        assert hi.confidence_score > lo.confidence_score


def test_analogue_index_roundtrip():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((500, 6)), columns=[f"f{i}" for i in range(6)])
    outcomes = pd.DataFrame({
        "label": rng.integers(0, 2, 500),
        "ret": rng.normal(0, 0.02, 500),
        "bars_held": rng.integers(1, 10, 500),
        "mae": -np.abs(rng.normal(0, 0.02, 500)),
        "mfe": np.abs(rng.normal(0, 0.02, 500)),
    })
    idx = AnalogueIndex(k=25, seed=1).fit(X, outcomes)
    rep = idx.query(X.iloc[0])
    assert rep.n == 25
    assert 0 <= rep.hit_rate <= 1
    assert rep.ret_quantiles["p10"] <= rep.ret_quantiles["p50"] <= rep.ret_quantiles["p90"]
    assert 0 < rep.similarity <= 1.0
    # a point far from the manifold must be less similar
    far = idx.query(pd.Series(8.0, index=X.columns))
    assert far.similarity < rep.similarity
