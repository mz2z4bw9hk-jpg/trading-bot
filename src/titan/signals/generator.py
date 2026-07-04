"""Signal generation: adaptive statistical gates + full decision packages.

A signal exists only if ALL of the following hold:

1. Calibrated probability clears the adaptive threshold — the break-even
   probability implied by the barrier geometry AND realistic costs, plus a
   configured margin, raised further in hostile regimes. The threshold is
   derived, not hand-tuned.
2. Expected value after costs is positive by at least the margin.
3. Ensemble disagreement is below the uncertainty ceiling.
4. The composite confidence score reaches at least grade B.
5. The risk engine assigns a positive size (crash regime alone zeroes it).

Everything else in this module is packaging: prices, sizes, analogue
statistics, and the evidence for and against the call.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.backtest.costs import CostModel
from titan.core.config import LabelConfig, RiskConfig, SignalConfig
from titan.core.log import get_logger
from titan.core.types import Regime, Side, TradeGrade, VolState
from titan.explain.evidence import LocalExplainer
from titan.features.rolling import atr as compute_atr
from titan.risk.sizing import atr_risk_size, fractional_kelly, vol_target_size
from titan.signals.analogues import AnalogueReport
from titan.signals.schema import Signal

logger = get_logger(__name__)

_HOSTILE_REGIMES = {Regime.BEAR, Regime.CORRECTION, Regime.DISTRIBUTION}
_ACCUMULATION_FEATURES = (
    "clv_accum_21", "obv_slope_21", "updown_vol_21", "dollar_vol_z_63", "vol_z_21",
)


class SignalGenerator:
    def __init__(
        self,
        signal_cfg: SignalConfig,
        label_cfg: LabelConfig,
        risk_cfg: RiskConfig,
        cost_model: CostModel,
        max_positions: int = 8,
    ) -> None:
        self._cfg = signal_cfg
        self._labels = label_cfg
        self._risk = risk_cfg
        self._costs = cost_model
        self._max_positions = max_positions

    # ------------------------------------------------------------------ #

    def adaptive_threshold(self, sigma: float, regime: Regime) -> tuple[float, float]:
        """(threshold, round_trip_cost). Break-even p plus margin, regime-tightened.

        With TP at +a and stop at -b, EV(p) = p·a − (1−p)·b − cost. Requiring
        EV ≥ margin gives p ≥ (b + cost + margin) / (a + b).
        """
        a = self._labels.tp_sigma * sigma
        b = self._labels.sl_sigma * sigma
        cost = self._costs.round_trip_cost_fraction(sigma)
        margin = self._cfg.ev_margin_bps / 1e4
        if regime in _HOSTILE_REGIMES:
            margin *= 2.0
        tau = (b + cost + margin) / max(a + b, 1e-9)
        return float(max(tau, self._cfg.min_probability)), cost

    # ------------------------------------------------------------------ #

    def generate(
        self,
        *,
        symbol: str,
        date: pd.Timestamp,
        probability: float,
        uncertainty: float,
        frame: pd.DataFrame,
        feature_row: pd.Series,
        regime: Regime,
        vol_state: VolState,
        regime_confidence: float,
        analogue: AnalogueReport | None = None,
        reliability: float = 1.0,
        explainer: LocalExplainer | None = None,
        model_version: str = "",
    ) -> Signal | None:
        """Build a signal for one (symbol, date) candidate, or return None.

        ``frame`` must contain history up to and including the decision bar.
        """
        if regime is Regime.CRASH:
            return None

        close = float(frame["close"].iloc[-1])
        sigma = float(
            np.log(frame["close"]).diff().ewm(span=self._labels.vol_span, adjust=False)
            .std().iloc[-1]
        )
        if not np.isfinite(sigma) or sigma <= 0:
            return None
        sigma = max(sigma, self._labels.min_vol_floor)

        tau, cost = self.adaptive_threshold(sigma, regime)
        if probability < tau:
            return None
        if uncertainty > self._cfg.max_uncertainty:
            return None

        a = self._labels.tp_sigma * sigma
        b = self._labels.sl_sigma * sigma
        ev = probability * a - (1.0 - probability) * b - cost
        if ev < self._cfg.ev_margin_bps / 1e4:
            return None

        similarity = analogue.similarity if analogue else 0.5
        confidence = self._confidence(probability, tau, uncertainty, similarity, reliability)
        grade = self._grade(confidence)
        if grade is None:
            return None

        # ---- sizing: minimum of three independent rules -----------------
        size = min(
            fractional_kelly(probability, a / b, self._risk.kelly_fraction,
                             self._risk.max_position_weight),
            vol_target_size(sigma * np.sqrt(252.0), self._risk.target_annual_vol,
                            self._max_positions, self._risk.max_position_weight),
            atr_risk_size(b, self._risk.risk_per_trade_pct, self._risk.max_position_weight),
        )
        if size <= 1e-4:
            return None

        # ---- prices ------------------------------------------------------
        atr_val = float(compute_atr(frame, 14).iloc[-1])
        stop = close * (1.0 - b)
        atr_stop = close - 2.0 * atr_val
        tps = [close * (1.0 + m * a) for m in (0.5, 1.0, 1.5)]
        limit_entry = close - 0.25 * atr_val
        entry_zone = (limit_entry, close + 0.10 * atr_val)

        # ---- evidence -----------------------------------------------------
        supporting: list[str] = []
        conflicting: list[str] = []
        top_effects: list[dict] = []
        if explainer is not None:
            effects = explainer.explain(feature_row, max_features=10)
            top_effects = [e.to_dict() for e in effects[:8]]
            for e in effects:
                if abs(e.delta_p) < 0.005:
                    continue
                desc = (
                    f"{e.feature} at {e.percentile:.0%} percentile "
                    f"({e.delta_p:+.1%} to probability)"
                )
                (supporting if e.delta_p > 0 else conflicting).append(desc)
        if analogue is not None:
            supporting.append(
                f"{analogue.n} nearest historical states: {analogue.hit_rate:.0%} hit rate, "
                f"median outcome {analogue.ret_quantiles['p50']:+.2%}"
            )
            if analogue.ret_quantiles["p25"] < -b:
                conflicting.append(
                    f"25th-percentile analogue outcome {analogue.ret_quantiles['p25']:+.2%} "
                    "is beyond the stop"
                )
        if regime in _HOSTILE_REGIMES:
            conflicting.append(f"hostile regime ({regime.value}): threshold tightened")
        if uncertainty > 0.6 * self._cfg.max_uncertainty:
            conflicting.append(f"elevated model disagreement ({uncertainty:.2f})")
        if reliability < 0.9:
            conflicting.append(f"data reliability {reliability:.2f}")

        institutional = self._institutional_score(feature_row, explainer)

        reasoning = (
            f"{symbol} long in {regime.value}/{vol_state.value}: calibrated "
            f"P(+{a:.1%} before -{b:.1%} within {self._labels.horizon_bars} bars) = "
            f"{probability:.0%} vs adaptive gate {tau:.0%}; EV after ~{cost:.2%} costs "
            f"= {ev:+.2%}. "
            + (
                f"Precedent: {analogue.n} analogues, {analogue.hit_rate:.0%} hits, "
                f"typical adverse excursion {analogue.mae_p75:.1%}. "
                if analogue
                else ""
            )
            + (f"Primary risk: {conflicting[0]}." if conflicting else "No material conflicting evidence.")
        )

        sig = Signal(
            symbol=symbol,
            date=date,
            side=Side.LONG,
            model_version=model_version,
            probability=probability,
            uncertainty=uncertainty,
            confidence_score=confidence,
            trade_grade=grade,
            threshold_used=tau,
            expected_return=ev,
            ev_after_costs=ev,
            cost_estimate=cost,
            risk_reward=a / b,
            outcome_quantiles=analogue.ret_quantiles if analogue else {},
            market_entry=close,
            optimal_limit_entry=limit_entry,
            entry_zone=entry_zone,
            stop_loss=stop,
            atr_stop=atr_stop,
            take_profit_levels=tps,
            position_size_fraction=size,
            risk_percentage=100.0 * size * b,
            expected_holding_bars=analogue.median_bars_held if analogue else self._labels.horizon_bars / 2,
            expected_volatility=sigma * np.sqrt(self._labels.horizon_bars),
            mae_estimate=analogue.mae_p75 if analogue else -b,
            mfe_estimate=analogue.mfe_p50 if analogue else a,
            market_regime=regime,
            vol_state=vol_state,
            regime_confidence=regime_confidence,
            data_reliability=reliability,
            historical_similarity=similarity,
            n_analogues=analogue.n if analogue else 0,
            institutional_score=institutional,
            supporting_evidence=supporting[:6],
            conflicting_evidence=conflicting[:6],
            top_features=top_effects,
            reasoning=reasoning,
        )
        return sig

    # ------------------------------------------------------------------ #

    def _confidence(
        self, p: float, tau: float, uncertainty: float, similarity: float, reliability: float
    ) -> float:
        """Composite 0-100. Transparent linear blend — no hidden magic.

        55% probability margin above the gate, 20% model agreement,
        15% historical precedent, 10% data quality.
        """
        p_margin = np.clip((p - tau) / 0.15, 0.0, 1.0)
        agreement = np.clip(1.0 - uncertainty / self._cfg.max_uncertainty, 0.0, 1.0)
        score = 100.0 * (
            0.55 * p_margin + 0.20 * agreement + 0.15 * similarity + 0.10 * reliability
        )
        return float(np.clip(score, 0.0, 100.0))

    def _grade(self, confidence: float) -> TradeGrade | None:
        t = self._cfg.grade_thresholds
        if confidence >= t.get("A+", 85.0):
            return TradeGrade.A_PLUS
        if confidence >= t.get("A", 75.0):
            return TradeGrade.A
        if confidence >= t.get("B+", 65.0):
            return TradeGrade.B_PLUS
        if confidence >= t.get("B", 55.0):
            return TradeGrade.B
        return None

    @staticmethod
    def _institutional_score(feature_row: pd.Series, explainer: LocalExplainer | None) -> float:
        """Accumulation-evidence composite from volume/participation features."""
        if explainer is None:
            return 50.0
        pcts = [
            explainer.percentile(f, float(feature_row[f]))
            for f in _ACCUMULATION_FEATURES
            if f in feature_row.index and not pd.isna(feature_row[f])
        ]
        return float(100.0 * np.mean(pcts)) if pcts else 50.0
