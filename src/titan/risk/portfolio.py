"""Portfolio-level risk engine.

Implements the backtester's :class:`RiskApprover` protocol: every candidate
entry is scaled (possibly to zero) against live portfolio state:

- portfolio heat cap: total open risk (sum of size × stop distance) bounded;
- per-position and per-sector concentration caps;
- correlation penalty: candidates highly correlated with the current book add
  less diversification than their size suggests, so they get less size;
- strategy drawdown throttle;
- regime multiplier: risk appetite follows the detected regime (zero in
  crash).

The same object serves live scanning and backtesting — one code path, no
sim-vs-prod drift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.backtest.engine import PortfolioSnapshot, TradePlan
from titan.core.config import RiskConfig
from titan.core.log import get_logger
from titan.core.types import Universe
from titan.risk.sizing import drawdown_throttle

logger = get_logger(__name__)


class RiskEngine:
    def __init__(
        self,
        cfg: RiskConfig,
        universe: Universe | None = None,
        returns: pd.DataFrame | None = None,
        regimes: pd.Series | None = None,
        corr_window: int = 63,
        var_window_bars: int = 252,
    ) -> None:
        """
        Parameters
        ----------
        returns:
            Wide (date x symbol) per-bar returns for correlation estimates.
        regimes:
            Regime label per date (strings from :class:`titan.core.types.Regime`).
        var_window_bars:
            Lookback for historical VaR/CVaR — one year expressed in bars, so
            it must track the run's timeframe rather than assume daily.
        """
        self._cfg = cfg
        self._universe = universe
        self._returns = returns
        self._regimes = regimes
        self._corr_window = corr_window
        self._var_window_bars = var_window_bars

    # ------------------------------------------------------------------ #

    def regime_multiplier(self, when: pd.Timestamp) -> float:
        if self._regimes is None:
            return 1.0
        try:
            idx = self._regimes.index.get_indexer([when], method="ffill")[0]
        except (KeyError, TypeError):
            return 1.0
        if idx < 0:
            return 1.0
        label = str(self._regimes.iloc[idx])
        return float(self._cfg.regime_multipliers.get(label, 0.5))

    def _avg_correlation_to_book(
        self, symbol: str, holdings: list[str], when: pd.Timestamp
    ) -> float:
        if self._returns is None or not holdings or symbol not in self._returns.columns:
            return 0.0
        window = self._returns.loc[:when].tail(self._corr_window)
        if len(window) < self._corr_window // 2:
            return 0.0
        cand = window[symbol]
        corrs = []
        for h in holdings:
            if h in window.columns:
                c = cand.corr(window[h])
                if np.isfinite(c):
                    corrs.append(c)
        return float(np.mean(corrs)) if corrs else 0.0

    # ------------------------------------------------------------------ #

    def approve(self, plan: TradePlan, snapshot: PortfolioSnapshot) -> float:
        """Return the approved size fraction for a candidate plan."""
        cfg = self._cfg
        size = min(plan.size_fraction, cfg.max_position_weight)

        # Regime appetite (crash -> 0).
        size *= self.regime_multiplier(plan.decision_date)
        if size <= 1e-6:
            return 0.0

        # Drawdown throttle.
        size *= drawdown_throttle(
            snapshot.strategy_drawdown, cfg.dd_throttle_start, cfg.dd_throttle_full
        )
        if size <= 1e-6:
            return 0.0

        # Portfolio heat: total open risk stays under the cap.
        entry_ref = plan.entry_ref if plan.entry_ref > 0 else plan.tp_price
        stop_frac = abs(entry_ref - plan.stop_price) / max(entry_ref, 1e-9)
        heat_cap = cfg.portfolio_heat_cap_pct / 100.0
        candidate_risk = size * stop_frac
        available = heat_cap - snapshot.open_risk_fraction
        if available <= 0:
            return 0.0
        if candidate_risk > available:
            size *= available / candidate_risk

        # Sector concentration.
        if self._universe is not None:
            sector = self._universe.sector_of(plan.symbol)
            current = snapshot.sector_weights.get(sector, 0.0)
            room = cfg.max_sector_weight - current
            if room <= 0:
                return 0.0
            size = min(size, room)

        # Correlation penalty: up to 50% haircut above the threshold.
        avg_corr = self._avg_correlation_to_book(
            plan.symbol, list(snapshot.symbol_weights), plan.decision_date
        )
        if avg_corr > cfg.correlation_penalty_threshold:
            excess = (avg_corr - cfg.correlation_penalty_threshold) / max(
                1.0 - cfg.correlation_penalty_threshold, 1e-9
            )
            size *= 1.0 - 0.5 * min(excess, 1.0)

        return float(max(size, 0.0))

    # ------------------------------------------------------------------ #

    def portfolio_var_cvar(
        self, weights: dict[str, float], when: pd.Timestamp, confidence: float | None = None
    ) -> tuple[float, float]:
        """Historical 1-bar VaR/CVaR of the current book (negative fractions)."""
        conf = confidence or self._cfg.var_confidence
        if self._returns is None or not weights:
            return 0.0, 0.0
        cols = [s for s in weights if s in self._returns.columns]
        if not cols:
            return 0.0, 0.0
        window = self._returns.loc[:when, cols].tail(self._var_window_bars)
        if len(window) < 60:
            return 0.0, 0.0
        w = np.array([weights[s] for s in cols])
        pnl = window.to_numpy() @ w
        var = float(np.quantile(pnl, 1.0 - conf))
        tail = pnl[pnl <= var]
        return var, float(tail.mean()) if len(tail) else var
