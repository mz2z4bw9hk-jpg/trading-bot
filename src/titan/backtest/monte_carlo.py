"""Monte Carlo robustness analysis.

Two complementary resampling schemes:

- **Stationary block bootstrap** (Politis & Romano 1994) on the strategy's
  daily returns: preserves short-range autocorrelation and volatility
  clustering while breaking the specific historical ordering. Confidence
  intervals on Sharpe/CAGR/MaxDD tell us how much of the backtest is path
  luck.
- **Trade-sequence resampling**: bootstraps the per-trade PnL distribution to
  estimate risk of ruin and drawdown quantiles under order-independence.

A strategy whose lower Sharpe CI touches zero is not a strategy; it is a
coin that came up heads once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from titan.backtest.metrics import TRADING_DAYS


def stationary_block_bootstrap(
    values: np.ndarray, n_sims: int, avg_block: float, seed: int
) -> np.ndarray:
    """(n_sims, len(values)) resamples with geometric block lengths."""
    n = len(values)
    if n < 10:
        raise ValueError("need at least 10 observations")
    rng = np.random.default_rng(seed)
    p_new_block = 1.0 / max(avg_block, 1.0)
    starts = rng.integers(0, n, size=(n_sims, n))
    new_block = rng.random((n_sims, n)) < p_new_block
    new_block[:, 0] = True
    idx = np.zeros((n_sims, n), dtype=int)
    for t in range(n):
        if t == 0:
            idx[:, 0] = starts[:, 0]
        else:
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(new_block[:, t], starts[:, t], cont)
    return values[idx]


@dataclass(slots=True)
class BootstrapReport:
    n_sims: int
    sharpe_ci: tuple[float, float]
    sharpe_median: float
    cagr_ci: tuple[float, float]
    max_dd_ci: tuple[float, float]  # (5th, 95th) pct of max drawdown (negative)
    p_sharpe_positive: float
    p_return_positive: float
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "n_sims": self.n_sims,
            "sharpe_median": round(self.sharpe_median, 4),
            "sharpe_ci_5_95": [round(v, 4) for v in self.sharpe_ci],
            "cagr_ci_5_95": [round(v, 6) for v in self.cagr_ci],
            "max_dd_ci_5_95": [round(v, 6) for v in self.max_dd_ci],
            "p_sharpe_positive": round(self.p_sharpe_positive, 4),
            "p_return_positive": round(self.p_return_positive, 4),
            **self.extras,
        }


def bootstrap_analysis(
    returns: pd.Series,
    n_sims: int = 1000,
    avg_block: float = 10.0,
    seed: int = 7,
    periods_per_year: float = TRADING_DAYS,
) -> BootstrapReport:
    r = returns.dropna().to_numpy()
    sims = stationary_block_bootstrap(r, n_sims=n_sims, avg_block=avg_block, seed=seed)

    means = sims.mean(axis=1)
    stds = sims.std(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpes = np.where(stds > 0, means / stds * np.sqrt(periods_per_year), 0.0)

    growth = np.cumprod(1.0 + sims, axis=1)
    years = sims.shape[1] / periods_per_year
    cagrs = growth[:, -1] ** (1.0 / years) - 1.0
    peaks = np.maximum.accumulate(growth, axis=1)
    max_dds = (growth / peaks - 1.0).min(axis=1)

    return BootstrapReport(
        n_sims=n_sims,
        sharpe_ci=(float(np.quantile(sharpes, 0.05)), float(np.quantile(sharpes, 0.95))),
        sharpe_median=float(np.median(sharpes)),
        cagr_ci=(float(np.quantile(cagrs, 0.05)), float(np.quantile(cagrs, 0.95))),
        max_dd_ci=(float(np.quantile(max_dds, 0.05)), float(np.quantile(max_dds, 0.95))),
        p_sharpe_positive=float((sharpes > 0).mean()),
        p_return_positive=float((cagrs > 0).mean()),
    )


def risk_of_ruin(
    trade_pnl_fractions: np.ndarray,
    trades_per_year: float,
    ruin_level: float = -0.30,
    horizon_years: float = 3.0,
    n_sims: int = 2000,
    seed: int = 7,
) -> dict:
    """P(cumulative equity path breaches ``ruin_level``) via trade resampling."""
    pnl = np.asarray(trade_pnl_fractions, dtype=float)
    if len(pnl) < 10:
        return {"risk_of_ruin": float("nan"), "note": "insufficient trades"}
    rng = np.random.default_rng(seed)
    n_trades = max(int(trades_per_year * horizon_years), 5)
    draws = rng.choice(pnl, size=(n_sims, n_trades), replace=True)
    equity = np.cumprod(1.0 + draws, axis=1)
    peaks = np.maximum.accumulate(equity, axis=1)
    dd = equity / peaks - 1.0
    ruined = (dd <= ruin_level).any(axis=1)
    return {
        "risk_of_ruin": float(ruined.mean()),
        "ruin_level": ruin_level,
        "horizon_years": horizon_years,
        "median_terminal": float(np.median(equity[:, -1])),
        "p5_terminal": float(np.quantile(equity[:, -1], 0.05)),
    }
