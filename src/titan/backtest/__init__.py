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


def __getattr__(name: str):  # PEP 562
    """Expose the walk-forward names without importing them at package load.

    ``walkforward`` pulls in titan.risk and titan.signals, both of which import
    titan.backtest.costs — which runs this module. Importing it eagerly here
    therefore made `import titan.scanner` (or titan.risk, or titan.signals)
    fail outright in a fresh interpreter, depending only on which subpackage
    the process happened to touch first. Resolving on first attribute access
    keeps `from titan.backtest import WalkForwardRunner` working while leaving
    the import graph acyclic.
    """
    if name in ("WalkForwardReport", "WalkForwardRunner"):
        from titan.backtest import walkforward

        return getattr(walkforward, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
