"""Performance and risk metrics, including PSR/DSR multiple-testing control.

Conventions: ``returns`` are per-bar simple returns of strategy equity;
``periods_per_year`` defaults to 252. Sharpe here is the per-year ratio with
zero risk-free rate (research convention; substitute a funding series in
production accounting if needed).

The deflated Sharpe ratio (Bailey & López de Prado, 2014) is a first-class
citizen: a platform that tries many configurations MUST discount its best
result by how many things it tried, or it is manufacturing overfitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy import stats

if TYPE_CHECKING:
    from titan.backtest.engine import TradeRecord

TRADING_DAYS = 252.0
EULER_GAMMA = 0.5772156649015329


def max_drawdown(equity: pd.Series) -> tuple[float, int]:
    """(max drawdown as a negative fraction, longest drawdown length in bars)."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0
    underwater = dd < 0
    longest = 0
    current = 0
    for flag in underwater:
        current = current + 1 if flag else 0
        longest = max(longest, current)
    return max_dd, longest


def sharpe_ratio(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    r = returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(r.mean() / r.std() * np.sqrt(periods_per_year))


def sortino_ratio(returns: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    r = returns.dropna()
    downside = r[r < 0]
    if len(r) < 2 or len(downside) == 0:
        return 0.0
    dd = float(np.sqrt(np.mean(np.square(downside))))
    if dd == 0:
        return 0.0
    return float(r.mean() / dd * np.sqrt(periods_per_year))


def cagr(equity: pd.Series, periods_per_year: float = TRADING_DAYS) -> float:
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    years = (len(equity) - 1) / periods_per_year
    if years <= 0:
        return 0.0
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0)


def var_cvar(returns: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Historical per-bar VaR and CVaR (negative fractions = losses)."""
    r = returns.dropna()
    if len(r) < 20:
        return 0.0, 0.0
    var = float(np.quantile(r, 1.0 - confidence))
    tail = r[r <= var]
    cvar = float(tail.mean()) if len(tail) else var
    return var, cvar


def probabilistic_sharpe_ratio(
    observed_sr: float,
    benchmark_sr: float,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """P(true SR > benchmark_sr) given sampling error and non-normality.

    All Sharpe ratios here are PER-PERIOD (not annualized). ``kurtosis`` is
    the raw fourth moment ratio (normal = 3).
    """
    if n_obs < 3:
        return 0.5
    denom = np.sqrt(
        max(1e-12, 1.0 - skew * observed_sr + (kurtosis - 1.0) / 4.0 * observed_sr**2)
    )
    z = (observed_sr - benchmark_sr) * np.sqrt(n_obs - 1.0) / denom
    return float(stats.norm.cdf(z))


def deflated_sharpe_ratio(
    observed_sr: float,
    sr_variance_across_trials: float,
    n_trials: int,
    n_obs: int,
    skew: float,
    kurtosis: float,
) -> float:
    """DSR: PSR against the expected maximum SR of ``n_trials`` unskilled trials.

    Bailey & López de Prado (2014). ``observed_sr`` per-period;
    ``sr_variance_across_trials`` is the variance of per-period SR estimates
    across everything that was tried before selecting this strategy.
    """
    n_trials = max(int(n_trials), 1)
    if n_trials == 1 or sr_variance_across_trials <= 0:
        sr_star = 0.0
    else:
        e_max_z = (1.0 - EULER_GAMMA) * stats.norm.ppf(1.0 - 1.0 / n_trials) + (
            EULER_GAMMA
        ) * stats.norm.ppf(1.0 - 1.0 / (n_trials * np.e))
        sr_star = float(np.sqrt(sr_variance_across_trials) * e_max_z)
    return probabilistic_sharpe_ratio(observed_sr, sr_star, n_obs, skew, kurtosis)


@dataclass(slots=True)
class PerfSummary:
    n_bars: int = 0
    n_trades: int = 0
    cagr: float = 0.0
    ann_vol: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_bars: int = 0
    profit_factor: float = 0.0
    expectancy: float = 0.0  # mean trade PnL as fraction of equity at entry
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    exposure: float = 0.0
    turnover_per_year: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    skew: float = 0.0
    kurtosis: float = 3.0
    tail_ratio: float = 1.0
    psr: float = 0.5
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        out = {k: getattr(self, k) for k in self.__dataclass_fields__ if k != "extras"}
        rounded = {
            k: (round(v, 6) if isinstance(v, float) else v) for k, v in out.items()
        }
        rounded.update(self.extras)
        return rounded


def summarize(
    equity: pd.Series,
    trades: list[TradeRecord] | None = None,
    exposure: pd.Series | None = None,
    periods_per_year: float = TRADING_DAYS,
    benchmark_sr_annual: float = 0.0,
) -> PerfSummary:
    returns = equity.pct_change().dropna()
    s = PerfSummary(n_bars=len(equity))
    if len(returns) < 2:
        return s

    s.cagr = cagr(equity, periods_per_year)
    s.ann_vol = float(returns.std() * np.sqrt(periods_per_year))
    s.sharpe = sharpe_ratio(returns, periods_per_year)
    s.sortino = sortino_ratio(returns, periods_per_year)
    s.max_drawdown, s.max_drawdown_bars = max_drawdown(equity)
    s.calmar = float(s.cagr / abs(s.max_drawdown)) if s.max_drawdown < 0 else 0.0
    s.var_95, s.cvar_95 = var_cvar(returns, 0.95)
    s.skew = float(stats.skew(returns))
    s.kurtosis = float(stats.kurtosis(returns, fisher=False))
    q95, q05 = np.quantile(returns, 0.95), np.quantile(returns, 0.05)
    s.tail_ratio = float(abs(q95) / abs(q05)) if q05 != 0 else float("inf")

    sr_period = float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
    s.psr = probabilistic_sharpe_ratio(
        sr_period,
        benchmark_sr_annual / np.sqrt(periods_per_year),
        len(returns),
        s.skew,
        s.kurtosis,
    )

    if trades:
        pnl = np.array([t.pnl_fraction for t in trades])
        s.n_trades = len(pnl)
        wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
        s.win_rate = float(len(wins) / len(pnl)) if len(pnl) else 0.0
        s.avg_win = float(wins.mean()) if len(wins) else 0.0
        s.avg_loss = float(losses.mean()) if len(losses) else 0.0
        gross_profit = float(wins.sum())
        gross_loss = float(-losses.sum())
        s.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
        s.expectancy = float(pnl.mean())
        years = len(equity) / periods_per_year
        notional_traded = sum(t.size_fraction * 2.0 for t in trades)  # entry + exit
        s.turnover_per_year = float(notional_traded / years) if years > 0 else 0.0

    if exposure is not None and len(exposure):
        s.exposure = float(exposure.mean())
    return s
