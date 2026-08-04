"""Margin trading: leverage, liquidation, and what it actually costs.

Spot equity is bounded below by zero and you can only lose what you posted. A
leveraged perpetual is a different instrument with a different failure mode:
the position carries more notional than the cash behind it, so an adverse move
of ``1/L`` wipes the margin out entirely — and the exchange closes the position
for you, at its price, before it gets there.

Three things follow, and this module exists to make all three explicit rather
than implied:

**Leverage multiplies risk, not just size.** A position sized to lose 0.4% of
equity at its stop loses 1.2% at 3x. There is no version of this where the
notional triples and the risk does not. Everything downstream — the order
card's risk column, the paper ledger, the portfolio heat cap — is fed the
*levered* number, because that is the number that is true.

**The stop has to sit inside the liquidation price.** A 3x position with a 40%
stop is not a 3x position with a 40% stop; it is a position that gets
liquidated at -32.8% and never reaches its stop. Rather than emit that order
and let the user discover it, :func:`plan` solves for the largest leverage
whose liquidation level stays a configured multiple beyond the stop, and uses
that. Wide stops therefore de-lever themselves automatically.

**Leverage is rented.** Perpetuals charge funding on notional for as long as
the position is open. It is small per bar and material over a swing hold, and
it scales with the notional the leverage created — so it belongs in the same
cost estimate that gates the trade, not in a footnote.

The convention here is isolated margin: each position posts ``notional /
leverage`` of its own and can lose no more than that. Cross margin, where one
position's loss eats another's collateral, is deliberately not modelled — it
would let a single trade liquidate the whole book, and nothing in this
platform's sizing logic is built to reason about that.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from titan.core.log import get_logger
from titan.core.timeframe import timeframe_hours
from titan.core.types import Side

if TYPE_CHECKING:
    from titan.core.config import LeverageConfig

logger = get_logger(__name__)

# Exchange maintenance margin for a mainstream perpetual at modest size.
# Real venues tier this by notional; a flat rate is conservative for the
# small-notional end where these orders live.
DEFAULT_MAINTENANCE_MARGIN_RATE = 0.005

# ~0.01% per 8h, the long-run resting rate on major perps, expressed daily.
DEFAULT_FUNDING_BPS_DAILY = 3.0


@dataclass(frozen=True, slots=True)
class LeverageTerms:
    """What an instrument is *allowed* to do, before its stop is consulted."""

    max_leverage: float = 1.0
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE
    funding_per_bar: float = 0.0     # fraction of notional, per bar held
    stop_buffer: float = 1.5         # liquidation must be this many stops away

    @property
    def enabled(self) -> bool:
        return self.max_leverage > 1.0

    @classmethod
    def spot(cls) -> LeverageTerms:
        """Cash trading: one dollar of position per dollar of equity."""
        return cls()

    @classmethod
    def resolve(
        cls, cfg: LeverageConfig, asset_class: str, timeframe: str
    ) -> LeverageTerms:
        """Terms for one asset class on one bar interval.

        Funding converts through the bar's wall-clock duration, not through the
        annualization constant: it accrues in real time, so a 3h bar rents the
        notional for 3h whether the calendar counts 252 days or 365.
        """
        max_lev = cfg.for_asset_class(asset_class)
        if max_lev <= 1.0:
            return cls.spot()
        return cls(
            max_leverage=max_lev,
            maintenance_margin_rate=cfg.maintenance_margin_rate,
            funding_per_bar=(cfg.funding_bps_daily / 1e4) * (timeframe_hours(timeframe) / 24.0),
            stop_buffer=cfg.stop_buffer,
        )


@dataclass(frozen=True, slots=True)
class LeveragePlan:
    """The levered form of an already-risk-sized position."""

    leverage: float
    notional_fraction: float          # position size as a fraction of equity
    margin_fraction: float            # cash actually posted
    risk_fraction_of_equity: float    # equity lost if the stop fills
    liquidation_price: float | None   # None when unlevered — spot cannot liquidate
    liquidation_distance: float | None  # fraction of entry, None when unlevered
    funding_cost: float               # fraction of notional over the expected hold

    @property
    def is_levered(self) -> bool:
        return self.leverage > 1.0


def liquidation_distance(leverage: float, maintenance_margin_rate: float) -> float:
    """Adverse move, as a fraction of entry, that exhausts the posted margin.

    Margin posted is ``N/L``; the exchange closes the position once equity in
    it falls to ``mmr·N``. Setting ``N/L − N·d = mmr·N`` gives ``d = 1/L −
    mmr``. At 1x that is ~99.5% — spot, for practical purposes unliquidatable,
    which is why :func:`plan` reports no liquidation level there at all.
    """
    if leverage <= 0:
        raise ValueError(f"leverage must be positive, got {leverage}")
    return max(1.0 / leverage - maintenance_margin_rate, 0.0)


def liquidation_price(
    entry: float, leverage: float, side: Side, maintenance_margin_rate: float
) -> float | None:
    """Price at which the position is force-closed. None when unlevered."""
    if leverage <= 1.0 or entry <= 0:
        return None
    d = liquidation_distance(leverage, maintenance_margin_rate)
    return entry * (1.0 - d) if side is Side.LONG else entry * (1.0 + d)


def safe_leverage(
    stop_distance: float,
    *,
    max_leverage: float,
    maintenance_margin_rate: float = DEFAULT_MAINTENANCE_MARGIN_RATE,
    stop_buffer: float = 1.5,
) -> float:
    """Largest leverage that keeps liquidation clear of the stop.

    Requiring ``1/L − mmr ≥ buffer·d_stop`` gives ``L ≤ 1/(buffer·d_stop +
    mmr)``. A tight stop leaves that constraint slack and the configured
    maximum binds; a wide one binds first and the position quietly de-levers.
    Clamped at 1.0, so a stop wide enough to make any leverage unsafe simply
    trades unlevered instead of being rejected.
    """
    if stop_distance <= 0:
        return 1.0
    ceiling = 1.0 / (stop_buffer * stop_distance + maintenance_margin_rate)
    return float(max(1.0, min(max_leverage, ceiling)))


def plan(
    *,
    base_size: float,
    entry: float,
    stop_distance: float,
    side: Side,
    holding_bars: float,
    terms: LeverageTerms,
) -> LeveragePlan:
    """Apply leverage to a position the risk engine has already sized.

    ``base_size`` is the unlevered weight — whatever ¼-Kelly ∧ vol-target ∧
    stop-risk agreed on. Leverage scales it into notional; the margin posted
    falls back out as ``notional / L``. The risk figure returned is computed on
    the levered notional, so a caller that reports it is reporting the truth
    about what this order can lose.
    """
    if not terms.enabled or stop_distance <= 0:
        return LeveragePlan(
            leverage=1.0,
            notional_fraction=base_size,
            margin_fraction=base_size,
            risk_fraction_of_equity=base_size * stop_distance,
            liquidation_price=None,
            liquidation_distance=None,
            funding_cost=0.0,
        )

    lev = safe_leverage(
        stop_distance,
        max_leverage=terms.max_leverage,
        maintenance_margin_rate=terms.maintenance_margin_rate,
        stop_buffer=terms.stop_buffer,
    )
    notional = base_size * lev
    d_liq = liquidation_distance(lev, terms.maintenance_margin_rate) if lev > 1.0 else None
    return LeveragePlan(
        leverage=lev,
        notional_fraction=notional,
        margin_fraction=notional / lev,
        risk_fraction_of_equity=notional * stop_distance,
        liquidation_price=liquidation_price(
            entry, lev, side, terms.maintenance_margin_rate
        ),
        liquidation_distance=d_liq,
        funding_cost=terms.funding_per_bar * max(holding_bars, 0.0),
    )


def liquidated(gross_return: float, leverage: float, maintenance_margin_rate: float) -> bool:
    """Did this round trip's path reach the liquidation level?

    Applied to a realized return in the ledger. Only an approximation of the
    live event — the exchange watches every tick and this sees one outcome per
    trade — but it catches the case that matters: a return worse than the
    margin can absorb, which without this check would post a loss larger than
    the cash the position ever had.
    """
    if leverage <= 1.0:
        return False
    return gross_return <= -liquidation_distance(leverage, maintenance_margin_rate)
