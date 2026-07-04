"""Backtesting: costs, engine, metrics, walk-forward, Monte Carlo."""

from titan.backtest.costs import CostModel
from titan.backtest.engine import (
    BacktestEngine,
    BacktestResult,
    PortfolioSnapshot,
    TradePlan,
    TradeRecord,
)
from titan.backtest.metrics import (
    PerfSummary,
    deflated_sharpe_ratio,
    probabilistic_sharpe_ratio,
    summarize,
)
from titan.backtest.monte_carlo import bootstrap_analysis, risk_of_ruin
from titan.backtest.walkforward import WalkForwardReport, WalkForwardRunner

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "CostModel",
    "PerfSummary",
    "PortfolioSnapshot",
    "TradePlan",
    "TradeRecord",
    "WalkForwardReport",
    "WalkForwardRunner",
    "bootstrap_analysis",
    "deflated_sharpe_ratio",
    "probabilistic_sharpe_ratio",
    "risk_of_ruin",
    "summarize",
]
