"""Transaction cost model.

Costs are the difference between a paper edge and a real one. The model has
three components per fill:

- commission: linear, in bps of notional;
- half-spread: crossing the book costs half the quoted spread;
- impact: scales with the instrument's daily volatility — a proxy for the
  depth/urgency trade-off that avoids needing order-book data. For sizes that
  are small relative to ADV this is conservative rather than optimistic.

Shorts additionally pay daily borrow. All rates come from config so cost
sensitivity analysis (docs/VALIDATION.md) can sweep them.
"""

from __future__ import annotations

from titan.core.config import CostConfig


class CostModel:
    def __init__(self, cfg: CostConfig) -> None:
        self._cfg = cfg

    def one_way_cost_fraction(self, daily_vol: float) -> float:
        """Fractional cost of a single fill (entry OR exit)."""
        commission = self._cfg.commission_bps / 1e4
        half_spread = self._cfg.spread_bps / 2e4
        impact = self._cfg.impact_coefficient * max(daily_vol, 0.0)
        return commission + half_spread + impact

    def round_trip_cost_fraction(self, daily_vol: float) -> float:
        return 2.0 * self.one_way_cost_fraction(daily_vol)

    def borrow_cost_fraction_per_bar(self) -> float:
        return self._cfg.borrow_bps_daily / 1e4

    def apply_entry(self, price: float, daily_vol: float, is_long: bool) -> float:
        """Fill price after slippage for an entry."""
        slip = self.one_way_cost_fraction(daily_vol)
        return price * (1.0 + slip) if is_long else price * (1.0 - slip)

    def apply_exit(self, price: float, daily_vol: float, is_long: bool) -> float:
        slip = self.one_way_cost_fraction(daily_vol)
        return price * (1.0 - slip) if is_long else price * (1.0 + slip)
