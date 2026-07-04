"""Signal schema: everything a desk needs to act on — or reject — a call.

A signal is not a prediction; it is a *decision package*: probabilities,
uncertainty, prices, sizes, risk, historical precedent, and the evidence for
and against. If any of that is missing, the signal should not exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from titan.core.types import Regime, Side, TradeGrade, VolState


@dataclass(slots=True)
class Signal:
    # identity ----------------------------------------------------------
    symbol: str
    date: pd.Timestamp
    side: Side
    model_version: str = ""

    # probability & confidence -------------------------------------------
    probability: float = 0.0          # calibrated P(tp before stop)
    uncertainty: float = 0.0          # ensemble disagreement (std of members)
    confidence_score: float = 0.0     # 0-100 composite
    trade_grade: TradeGrade = TradeGrade.B
    threshold_used: float = 0.0       # adaptive gate this signal cleared

    # economics ------------------------------------------------------------
    expected_return: float = 0.0      # net of costs, fraction, over the horizon
    ev_after_costs: float = 0.0       # alias of the gating quantity
    cost_estimate: float = 0.0        # round-trip fraction
    risk_reward: float = 0.0
    outcome_quantiles: dict[str, float] = field(default_factory=dict)  # p10..p90

    # prices ------------------------------------------------------------
    market_entry: float = 0.0         # expected fill (near decision close)
    optimal_limit_entry: float = 0.0
    entry_zone: tuple[float, float] = (0.0, 0.0)
    stop_loss: float = 0.0            # sigma-based stop used by plans
    atr_stop: float = 0.0
    take_profit_levels: list[float] = field(default_factory=list)

    # sizing / risk ----------------------------------------------------------
    position_size_fraction: float = 0.0
    risk_percentage: float = 0.0      # equity at risk if stopped, in %
    expected_holding_bars: float = 0.0
    expected_volatility: float = 0.0  # over holding horizon, fraction
    mae_estimate: float = 0.0         # typical adverse excursion (fraction)
    mfe_estimate: float = 0.0

    # context ----------------------------------------------------------
    market_regime: Regime = Regime.RANGE
    vol_state: VolState = VolState.NORMAL
    regime_confidence: float = 0.0
    data_reliability: float = 1.0
    historical_similarity: float = 0.0   # 0-1: how close the analogues are
    n_analogues: int = 0
    institutional_score: float = 0.0     # 0-100 accumulation-evidence composite

    # explanation ----------------------------------------------------------
    supporting_evidence: list[str] = field(default_factory=list)
    conflicting_evidence: list[str] = field(default_factory=list)
    top_features: list[dict] = field(default_factory=list)  # name/value/impact
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "date": str(self.date.date()),
            "side": self.side.value,
            "model_version": self.model_version,
            "probability": round(self.probability, 4),
            "uncertainty": round(self.uncertainty, 4),
            "confidence_score": round(self.confidence_score, 1),
            "trade_grade": self.trade_grade.value,
            "threshold_used": round(self.threshold_used, 4),
            "expected_return": round(self.expected_return, 5),
            "ev_after_costs": round(self.ev_after_costs, 5),
            "cost_estimate": round(self.cost_estimate, 5),
            "risk_reward": round(self.risk_reward, 3),
            "outcome_quantiles": {k: round(v, 5) for k, v in self.outcome_quantiles.items()},
            "market_entry": round(self.market_entry, 4),
            "optimal_limit_entry": round(self.optimal_limit_entry, 4),
            "entry_zone": [round(p, 4) for p in self.entry_zone],
            "stop_loss": round(self.stop_loss, 4),
            "atr_stop": round(self.atr_stop, 4),
            "take_profit_levels": [round(p, 4) for p in self.take_profit_levels],
            "position_size_fraction": round(self.position_size_fraction, 4),
            "risk_percentage": round(self.risk_percentage, 3),
            "expected_holding_bars": round(self.expected_holding_bars, 1),
            "expected_volatility": round(self.expected_volatility, 4),
            "mae_estimate": round(self.mae_estimate, 4),
            "mfe_estimate": round(self.mfe_estimate, 4),
            "market_regime": self.market_regime.value,
            "vol_state": self.vol_state.value,
            "regime_confidence": round(self.regime_confidence, 3),
            "data_reliability": round(self.data_reliability, 3),
            "historical_similarity": round(self.historical_similarity, 3),
            "n_analogues": self.n_analogues,
            "institutional_score": round(self.institutional_score, 1),
            "supporting_evidence": list(self.supporting_evidence),
            "conflicting_evidence": list(self.conflicting_evidence),
            "top_features": self.top_features,
            "reasoning": self.reasoning,
        }
