"""Champion/challenger statistical comparison.

A challenger replaces production only if it is *statistically* better on
overlapping out-of-sample history — not merely luckier. The test is a paired
stationary-block bootstrap on the per-bar return differential, which respects
autocorrelation and volatility clustering and asks the only question that
matters: "how often would a difference this large appear if the two
strategies were equally good?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.backtest.monte_carlo import stationary_block_bootstrap
from titan.core.log import get_logger
from titan.core.timeframe import TRADING_DAYS_PER_YEAR

logger = get_logger(__name__)


def compare_returns(
    challenger: pd.Series,
    production: pd.Series,
    n_sims: int = 2000,
    avg_block: float = 10.0,
    seed: int = 7,
    periods_per_year: float = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Paired bootstrap of the challenger-minus-production return spread.

    Returns the one-sided p-value for H0: challenger is not better.
    """
    joined = pd.concat(
        {"challenger": challenger, "production": production}, axis=1
    ).dropna()
    if len(joined) < 60:
        return {
            "n_obs": len(joined),
            "verdict": "insufficient_overlap",
            "p_value": float("nan"),
        }
    diff = (joined["challenger"] - joined["production"]).to_numpy()
    observed = float(diff.mean())

    sims = stationary_block_bootstrap(diff - observed, n_sims=n_sims, avg_block=avg_block, seed=seed)
    null_means = sims.mean(axis=1)
    p_value = float((null_means >= observed).mean())

    def _sharpe(x: pd.Series) -> float:
        return (
            float(x.mean() / x.std() * np.sqrt(periods_per_year)) if x.std() > 0 else 0.0
        )

    return {
        "n_obs": len(joined),
        "mean_daily_spread": observed,
        "spread_t_stat": float(observed / (diff.std() / np.sqrt(len(diff)))) if diff.std() > 0 else 0.0,
        "challenger_sharpe": _sharpe(joined["challenger"]),
        "production_sharpe": _sharpe(joined["production"]),
        "p_value": p_value,
    }


def promotion_gate(
    comparison: dict,
    challenger_summary: dict,
    production_summary: dict,
    p_value_required: float = 0.05,
) -> tuple[bool, list[str]]:
    """Decide promotion. ALL conditions must hold; reasons list the failures.

    1. Paired bootstrap p-value below the configured level.
    2. Challenger Sharpe strictly above production's.
    3. Challenger max drawdown not more than 1.25x production's.
    4. Enough overlapping history to mean anything.
    """
    reasons: list[str] = []
    p = comparison.get("p_value", float("nan"))
    if not np.isfinite(p):
        reasons.append("insufficient overlapping out-of-sample history")
    elif p > p_value_required:
        reasons.append(f"spread not significant (p={p:.3f} > {p_value_required})")
    if comparison.get("challenger_sharpe", 0.0) <= comparison.get("production_sharpe", 0.0):
        reasons.append("challenger Sharpe does not exceed production")
    ch_dd = abs(challenger_summary.get("max_drawdown", 0.0))
    pr_dd = abs(production_summary.get("max_drawdown", 1e-9))
    if pr_dd > 0 and ch_dd > 1.25 * pr_dd:
        reasons.append(f"challenger drawdown {ch_dd:.1%} exceeds 1.25x production {pr_dd:.1%}")
    approved = not reasons
    logger.info("promotion gate: %s%s", "APPROVED" if approved else "REJECTED",
                "" if approved else f" ({'; '.join(reasons)})")
    return approved, reasons
