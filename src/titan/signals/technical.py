"""Rule-based swing-trading setups: a second, independent source of orders.

The rest of the platform asks one question — *what is the calibrated
probability that price hits +tp before -sl?* — and refuses to trade when the
answer is not confidently above the cost-derived break-even. On a universe
where the model scores AUC ~0.51 that gate correctly emits nothing, forever.

These setups ask a different question. They are classic price-structure
patterns evaluated by rule: is this a breakout, a pullback in an uptrend, a
trend change, an oversold bounce in a bull market, a momentum continuation.
No model, no probability, no claim of edge — just the conditions written down
explicitly and checked on the bar.

WHAT THIS BUYS AND WHAT IT COSTS
--------------------------------
Buys: orders on days the ML gate is silent, with entries, stops and targets
taken from actual price structure (swing lows, ATR, R-multiples) rather than
from a model's sigma. Every one is labelled with the setup that produced it.

Costs: these are NOT validated alpha. The walk-forward, the Venn-ABERS bands,
the deflated Sharpe — none of that machinery applies to a rule fired on a
chart. A breakout setup has no out-of-sample evidence behind it here; it has a
definition. The paper account is what settles the question: every technical
order is logged and graded like any other, so after enough of them the ledger
says whether the rules made money. Until then they are a hypothesis being
tested with simulated money, which is exactly what they should be.

CAUSALITY
---------
Every setup reads only ``frame`` up to and including its last row, and the
last row is the decision bar. Entry is the decision close (the engine fills at
the next open, as everywhere else). A test recomputes each setup on truncated
history and requires identical output.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from titan.core.log import get_logger
from titan.core.types import Side
from titan.features import rolling as R
from titan.risk.leverage import LeverageTerms

if TYPE_CHECKING:  # avoid a cycle: schema imports nothing from here
    from titan.signals.schema import Signal

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class TechnicalSetup:
    """A fired rule, with the trade it implies."""

    name: str
    side: Side
    entry: float
    stop: float
    targets: list[float]
    strength: float               # 0-1, for ranking when more fire than wanted
    rationale: str
    evidence: list[str] = field(default_factory=list)

    @property
    def risk_fraction(self) -> float:
        """Distance to the stop as a fraction of entry — the R in R-multiples."""
        return abs(self.entry - self.stop) / self.entry if self.entry > 0 else 0.0

    @property
    def risk_reward(self) -> float:
        if not self.targets or self.risk_fraction <= 0:
            return 0.0
        reward = abs(self.targets[min(1, len(self.targets) - 1)] - self.entry) / self.entry
        return reward / self.risk_fraction


def _sma(s: pd.Series, w: int) -> pd.Series:
    return s.rolling(w, min_periods=w).mean()


def _ema(s: pd.Series, w: int) -> pd.Series:
    return s.ewm(span=w, adjust=False, min_periods=w).mean()


def _targets_from_r(entry: float, stop: float, multiples=(1.0, 2.0, 3.0)) -> list[float]:
    """Take-profits at R multiples of the stop distance.

    Structure decides the stop; the targets follow from it. This keeps reward
    commensurate with the risk actually being taken rather than with a fixed
    percentage that means something different on every instrument.
    """
    r = entry - stop
    return [entry + m * r for m in multiples]


def _swing_low(frame: pd.DataFrame, lookback: int) -> float:
    return float(frame["low"].iloc[-lookback:].min())


def _enough(frame: pd.DataFrame, bars: int) -> bool:
    return len(frame) >= bars and frame["close"].iloc[-bars:].notna().all()


# --------------------------------------------------------------- setups ----


def donchian_breakout(frame: pd.DataFrame, lookback: int = 20) -> TechnicalSetup | None:
    """Close at a new N-bar high, confirmed by volume.

    The canonical trend-following entry. Stop under the breakout base (the
    lowest low of the channel's second half), so the trade is wrong exactly
    when price falls back into the range it just left.
    """
    if not _enough(frame, lookback + 25):
        return None
    close = frame["close"]
    prior_high = float(close.iloc[-(lookback + 1):-1].max())
    entry = float(close.iloc[-1])
    if entry <= prior_high:
        return None

    vol = frame["volume"]
    avg_vol = float(vol.iloc[-21:-1].mean())
    if avg_vol <= 0 or float(vol.iloc[-1]) < 1.2 * avg_vol:
        return None

    stop = _swing_low(frame, lookback // 2)
    if stop >= entry:
        return None
    atr = float(R.atr(frame, 14).iloc[-1])
    excess = (entry - prior_high) / atr if atr > 0 else 0.0
    return TechnicalSetup(
        name="donchian_breakout",
        side=Side.LONG,
        entry=entry,
        stop=stop,
        targets=_targets_from_r(entry, stop),
        strength=float(np.clip(excess, 0.0, 1.0)),
        rationale=(
            f"Closed at a new {lookback}-bar high ({entry:.4g} vs {prior_high:.4g}) "
            f"on {float(vol.iloc[-1]) / avg_vol:.1f}x average volume."
        ),
        evidence=[
            f"breakout {excess:.2f} ATR above the prior {lookback}-bar high",
            f"volume {float(vol.iloc[-1]) / avg_vol:.1f}x the 20-bar average",
        ],
    )


def pullback_in_uptrend(frame: pd.DataFrame) -> TechnicalSetup | None:
    """Buy the dip to the 20MA inside an established uptrend.

    The pullback is defined by price returning to its own short moving average,
    not by an RSI threshold. Codifying it as "RSI(14) below 40" reads like the
    same idea but is nearly the opposite condition: by the time a 14-period RSI
    is under 40 the price is usually beneath the 50MA, so the trend filter and
    the pullback filter almost never hold at once and the rule never fires.
    Distance to the 20MA measures the same pullback while the trend survives.
    """
    if not _enough(frame, 220):
        return None
    close = frame["close"]
    sma20, sma50, sma200 = _sma(close, 20), _sma(close, 50), _sma(close, 200)
    if not (close.iloc[-1] > sma50.iloc[-1] > sma200.iloc[-1]):
        return None
    if not sma50.iloc[-1] > sma50.iloc[-6]:              # the 50 must be rising
        return None

    atr = float(R.atr(frame, 14).iloc[-1])
    if atr <= 0:
        return None
    # Came back to the 20MA within the last few bars...
    distance = (close.iloc[-5:] - sma20.iloc[-5:]).abs() / atr
    touched = bool((distance < 1.0).any())
    # ...cooled off, and is turning back up.
    rsi = R.wilder_rsi(close, 14)
    cooled = bool((rsi.iloc[-6:-1] < 55).any())
    turning = bool(close.iloc[-1] > close.iloc[-2] and rsi.iloc[-1] > rsi.iloc[-2])
    if not (touched and cooled and turning):
        return None

    entry = float(close.iloc[-1])
    stop = min(_swing_low(frame, 10), float(sma50.iloc[-1]) * 0.995)
    if stop >= entry:
        return None
    return TechnicalSetup(
        name="pullback_in_uptrend",
        side=Side.LONG,
        entry=entry,
        stop=stop,
        targets=_targets_from_r(entry, stop),
        strength=float(np.clip(1.0 - float(distance.min()), 0.0, 1.0)),
        rationale=(
            "Pullback inside an uptrend: price above a rising 50MA above the 200MA, "
            f"back to within {float(distance.min()):.2f} ATR of the 20MA and turning up."
        ),
        evidence=[
            "close > 50MA > 200MA with the 50MA rising",
            f"pulled back to {float(distance.min()):.2f} ATR of the 20MA",
            f"RSI(14) cooled to {float(rsi.iloc[-6:-1].min()):.0f} and is ticking up",
        ],
    )


def ma_cross(frame: pd.DataFrame) -> TechnicalSetup | None:
    """20MA crosses above the 50MA while price holds above the 200MA."""
    if not _enough(frame, 220):
        return None
    close = frame["close"]
    fast, slow, long_ma = _sma(close, 20), _sma(close, 50), _sma(close, 200)
    crossed = fast.iloc[-1] > slow.iloc[-1] and fast.iloc[-2] <= slow.iloc[-2]
    if not (crossed and close.iloc[-1] > long_ma.iloc[-1]):
        return None

    entry = float(close.iloc[-1])
    atr = float(R.atr(frame, 14).iloc[-1])
    stop = min(_swing_low(frame, 15), entry - 2.0 * atr)
    if stop >= entry or atr <= 0:
        return None
    separation = float((fast.iloc[-1] - slow.iloc[-1]) / atr)
    return TechnicalSetup(
        name="ma_cross",
        side=Side.LONG,
        entry=entry,
        stop=stop,
        targets=_targets_from_r(entry, stop),
        strength=float(np.clip(separation, 0.0, 1.0)),
        rationale="20MA crossed above the 50MA with price above the 200MA.",
        evidence=["20/50 MA bullish cross on this bar", "close above the 200MA"],
    )


def oversold_bounce(frame: pd.DataFrame) -> TechnicalSetup | None:
    """Short-term washout inside a long-term uptrend (RSI(2) < 10 above the 200MA)."""
    if not _enough(frame, 220):
        return None
    close = frame["close"]
    sma200 = _sma(close, 200)
    if not close.iloc[-1] > sma200.iloc[-1]:
        return None

    rsi2 = R.wilder_rsi(close, 2)
    if not rsi2.iloc[-1] < 10:
        return None

    entry = float(close.iloc[-1])
    atr = float(R.atr(frame, 14).iloc[-1])
    stop = min(_swing_low(frame, 5), entry - 1.5 * atr)
    if stop >= entry or atr <= 0:
        return None
    return TechnicalSetup(
        name="oversold_bounce",
        side=Side.LONG,
        entry=entry,
        stop=stop,
        # Mean reversion is a shorter trade than a breakout: nearer targets.
        targets=_targets_from_r(entry, stop, multiples=(0.75, 1.5, 2.5)),
        strength=float(np.clip((10.0 - float(rsi2.iloc[-1])) / 10.0, 0.0, 1.0)),
        rationale=(
            f"RSI(2) at {float(rsi2.iloc[-1]):.0f} — a short-term washout with "
            "price still above its 200MA."
        ),
        evidence=[f"RSI(2) = {float(rsi2.iloc[-1]):.0f} (below 10)",
                  "long-term uptrend intact: close above the 200MA"],
    )


def macd_momentum(frame: pd.DataFrame) -> TechnicalSetup | None:
    """MACD crosses its signal line above zero — continuation, not reversal."""
    if not _enough(frame, 220):
        return None
    close = frame["close"]
    macd = _ema(close, 12) - _ema(close, 26)
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    crossed = macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]
    if not (crossed and macd.iloc[-1] > 0 and close.iloc[-1] > _sma(close, 200).iloc[-1]):
        return None

    entry = float(close.iloc[-1])
    atr = float(R.atr(frame, 14).iloc[-1])
    stop = min(_swing_low(frame, 10), entry - 2.0 * atr)
    if stop >= entry or atr <= 0:
        return None
    return TechnicalSetup(
        name="macd_momentum",
        side=Side.LONG,
        entry=entry,
        stop=stop,
        targets=_targets_from_r(entry, stop),
        strength=float(np.clip(float(macd.iloc[-1] - signal.iloc[-1]) / atr, 0.0, 1.0)),
        rationale="MACD crossed above its signal line while above zero, in an uptrend.",
        evidence=["MACD/signal bullish cross above zero", "close above the 200MA"],
    )


SETUPS: dict[str, Callable[[pd.DataFrame], TechnicalSetup | None]] = {
    "donchian_breakout": donchian_breakout,
    "pullback_in_uptrend": pullback_in_uptrend,
    "ma_cross": ma_cross,
    "oversold_bounce": oversold_bounce,
    "macd_momentum": macd_momentum,
}


def detect(
    frame: pd.DataFrame,
    setups: list[str] | None = None,
    *,
    min_risk_reward: float = 1.5,
) -> list[TechnicalSetup]:
    """Every enabled setup that fires on the last bar of ``frame``.

    Setups whose second target does not clear ``min_risk_reward`` are dropped:
    a rule that fires but pays less than it risks is not a trade.
    """
    names = setups if setups is not None else list(SETUPS)
    out: list[TechnicalSetup] = []
    for name in names:
        fn = SETUPS.get(name)
        if fn is None:
            raise ValueError(f"unknown technical setup {name!r}; have {sorted(SETUPS)}")
        try:
            hit = fn(frame)
        except (ValueError, IndexError, KeyError):  # a short or ragged frame
            continue
        if hit is None:
            continue
        if hit.risk_reward < min_risk_reward or hit.risk_fraction <= 0:
            continue
        out.append(hit)
    return out


# ------------------------------------------------------- to a full order ----


def to_signal(
    setup: TechnicalSetup,
    *,
    symbol: str,
    date: pd.Timestamp,
    frame: pd.DataFrame,
    risk_cfg,
    cost_model,
    regime,
    vol_state,
    regime_confidence: float = 0.0,
    reliability: float = 1.0,
    asset_class: str = "equity",
    leverage_terms: LeverageTerms | None = None,
) -> Signal | None:
    """Package a fired rule as the same Signal the rest of the platform speaks.

    Sizing is fixed-fractional off the rule's own stop — risk_per_trade_pct of
    equity if the stop is hit — capped by max_position_weight and scaled by the
    regime multiplier the risk engine would apply. No Kelly term: Kelly needs a
    probability, and a rule does not produce one. Reporting a made-up
    ``probability`` here would put a number on the order card that nothing
    computed, so it stays at zero and the card shows the setup instead.

    ``leverage_terms`` then scales that unlevered weight into notional. The
    reported risk is recomputed on the levered notional, and funding for the
    expected hold joins the cost the setup has to clear — leverage that cannot
    pay its own rent out of the move it is predicting is not worth taking.
    """
    from titan.core.types import TradeGrade
    from titan.risk.leverage import plan as leverage_plan
    from titan.risk.sizing import atr_risk_size
    from titan.signals.schema import Signal

    entry = setup.entry
    risk_fraction = setup.risk_fraction
    multiplier = risk_cfg.regime_multipliers.get(regime.value, 1.0)
    size = atr_risk_size(
        risk_fraction, risk_cfg.risk_per_trade_pct, risk_cfg.max_position_weight
    ) * multiplier
    if size <= 1e-4:
        return None

    terms = leverage_terms or LeverageTerms.spot()
    holding_bars = 10.0
    lev = leverage_plan(
        base_size=size,
        entry=entry,
        stop_distance=risk_fraction,
        side=setup.side,
        holding_bars=holding_bars,
        terms=terms,
    )

    sigma = float(
        np.log(frame["close"]).diff().ewm(span=21, adjust=False).std().iloc[-1]
    )
    cost = cost_model.round_trip_cost_fraction(sigma if np.isfinite(sigma) else 0.01)
    cost += lev.funding_cost
    atr = float(R.atr(frame, 14).iloc[-1])
    reward = (setup.targets[min(1, len(setup.targets) - 1)] - entry) / entry
    if reward <= cost:
        return None

    # Grade from the rule's own quality, not from a probability it never had.
    score = 50.0 + 25.0 * setup.strength + 10.0 * min(setup.risk_reward / 3.0, 1.0)
    grade = (
        TradeGrade.A_PLUS if score >= 80 else
        TradeGrade.A if score >= 70 else
        TradeGrade.B_PLUS if score >= 60 else TradeGrade.B
    )

    evidence = list(setup.evidence)
    conflicting = [
        "rule-based setup: no out-of-sample probability behind this order",
    ]
    if lev.is_levered:
        evidence.append(
            f"{lev.leverage:.1f}x margin: {lev.margin_fraction:.1%} of equity posted "
            f"for {lev.notional_fraction:.1%} of notional"
        )
        conflicting.append(
            f"leveraged {lev.leverage:.1f}x — the stop now costs "
            f"{100.0 * lev.risk_fraction_of_equity:.2f}% of equity, and liquidation "
            f"sits at {lev.liquidation_price:.6g} "
            f"({lev.liquidation_distance:.1%} against the entry)"
        )

    return Signal(
        symbol=symbol,
        date=date,
        side=setup.side,
        model_version="technical",
        source=f"technical:{setup.name}",
        asset_class=asset_class,
        probability=0.0,
        uncertainty=0.0,
        confidence_score=round(score, 1),
        trade_grade=grade,
        threshold_used=0.0,
        expected_return=reward - cost,
        ev_after_costs=reward - cost,
        cost_estimate=cost,
        risk_reward=setup.risk_reward,
        market_entry=entry,
        optimal_limit_entry=entry - 0.25 * atr if atr > 0 else entry,
        entry_zone=(entry - 0.25 * atr if atr > 0 else entry, entry + 0.10 * atr),
        stop_loss=setup.stop,
        atr_stop=entry - 2.0 * atr if atr > 0 else setup.stop,
        take_profit_levels=list(setup.targets),
        position_size_fraction=lev.notional_fraction,
        risk_percentage=100.0 * lev.risk_fraction_of_equity,
        leverage=lev.leverage,
        margin_fraction=lev.margin_fraction,
        liquidation_price=lev.liquidation_price,
        funding_cost=lev.funding_cost,
        expected_holding_bars=holding_bars,
        expected_volatility=abs(sigma) if np.isfinite(sigma) else 0.0,
        market_regime=regime,
        vol_state=vol_state,
        regime_confidence=regime_confidence,
        data_reliability=reliability,
        supporting_evidence=evidence,
        conflicting_evidence=conflicting,
        reasoning=setup.rationale + (
            f" Traded at {lev.leverage:.1f}x margin." if lev.is_levered else ""
        ),
    )
