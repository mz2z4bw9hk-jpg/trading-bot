"""Position sizing primitives.

The platform sizes every position as the MINIMUM of three independent rules —
edge-based (fractional Kelly), volatility-based (vol targeting) and
loss-based (fixed fractional risk at the stop). Each rule alone has a known
failure mode; the minimum inherits none of them:

- Kelly with a mis-estimated edge over-bets catastrophically → we take a
  configured fraction (default 25%) of Kelly and cap it.
- Vol targeting ignores where the stop is → the ATR/stop rule bounds the loss
  if the stop is hit.
- Fixed fractional risk ignores how volatile the asset is between entry and
  stop → vol targeting bounds day-to-day portfolio swing.
"""

from __future__ import annotations

import numpy as np


def fractional_kelly(
    p: float, payoff_ratio: float, kelly_fraction: float, cap: float
) -> float:
    """Fraction of equity from a binary-outcome Kelly criterion.

    f* = (p·b − (1−p)) / b, scaled by ``kelly_fraction``, clipped to [0, cap].
    ``payoff_ratio`` b is the win/loss magnitude ratio (tp distance / stop
    distance under the barrier geometry).
    """
    if not 0.0 < p < 1.0 or payoff_ratio <= 0:
        return 0.0
    f_star = (p * payoff_ratio - (1.0 - p)) / payoff_ratio
    return float(np.clip(f_star * kelly_fraction, 0.0, cap))


def vol_target_size(
    asset_annual_vol: float,
    target_portfolio_vol: float,
    max_concurrent_positions: int,
    cap: float,
) -> float:
    """Size so this position's vol contribution fits the portfolio budget.

    Under a conservative equal-and-correlated assumption the per-position
    budget is ``target_vol / sqrt(N)`` with N the configured maximum book
    size; weight = budget / asset_vol.
    """
    if asset_annual_vol <= 1e-6:
        return 0.0
    budget = target_portfolio_vol / np.sqrt(max(max_concurrent_positions, 1))
    return float(np.clip(budget / asset_annual_vol, 0.0, cap))


def atr_risk_size(stop_distance_fraction: float, risk_per_trade_pct: float, cap: float) -> float:
    """Fixed fractional risk: lose at most ``risk_per_trade_pct`` of equity at the stop."""
    if stop_distance_fraction <= 1e-6:
        return 0.0
    return float(np.clip((risk_per_trade_pct / 100.0) / stop_distance_fraction, 0.0, cap))


def drawdown_throttle(current_drawdown: float, start: float, full: float) -> float:
    """De-risking multiplier in [0, 1] as strategy drawdown deepens.

    1.0 until ``-start`` drawdown, linearly falling to 0.0 at ``-full``. Keeps
    the platform solvent while models degrade faster than monitoring catches.
    """
    dd = abs(min(current_drawdown, 0.0))
    if dd <= start:
        return 1.0
    if dd >= full:
        return 0.0
    return float(1.0 - (dd - start) / (full - start))
