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

from collections.abc import Mapping

import numpy as np
import pandas as pd

from titan.backtest.engine import PortfolioSnapshot, TradePlan
from titan.core.config import RiskConfig
from titan.core.log import get_logger
from titan.core.timeframe import INTRADAY_TIMEFRAMES
from titan.core.types import Universe
from titan.risk.sizing import drawdown_throttle

logger = get_logger(__name__)


def _match_tz(when: pd.Timestamp, index: pd.Index) -> pd.Timestamp:
    """Align a timestamp's tz-awareness to an index's, so slicing is legal."""
    aware = getattr(index, "tz", None) is not None
    if aware and when.tz is None:
        return when.tz_localize("UTC")
    if not aware and when.tz is not None:
        return when.tz_convert("UTC").tz_localize(None)
    return when


def build_returns_matrix(
    frames: Mapping[str, pd.DataFrame], timeframe: str = "1d"
) -> pd.DataFrame:
    """Wide (date x symbol) returns on a COMMON calendar grid.

    Three things have to happen here, and skipping any one of them silently
    breaks the correlation estimate rather than failing.

    **Difference before aligning.** Aligning closes on a union index and
    calling ``pct_change`` afterwards makes every post-gap return NaN — an
    equity's Monday reads back to a NaN weekend row. That deletes precisely
    the gap returns, which is where correlated names move together hardest.

    **Put every index in UTC.** yfinance stamps a US equity's daily bar in
    ``America/New_York`` and a coin's in UTC, so the same trading day arrives
    as 04:00Z and 00:00Z. Concatenating those merges nothing: the union index
    comes out roughly twice as long as it should, every ``tail(window)`` covers
    half the history it claims, and no equity/crypto pair shares a single
    observation. Cross-asset correlation was not weak, it was undefined.

    **Normalise to the day when the timeframe is daily or coarser.** UTC alone
    does not fix the above — 04:00Z and 00:00Z are still different instants. A
    daily bar denotes a session, not a moment, so the grid is the date.
    Intraday bars genuinely are moments and align on the hour once in UTC, so
    they are left alone.
    """
    daily_or_coarser = timeframe not in INTRADAY_TIMEFRAMES
    series: dict[str, pd.Series] = {}
    for symbol, frame in frames.items():
        r = frame["close"].pct_change()
        idx = pd.DatetimeIndex(r.index)
        idx = idx.tz_localize("UTC") if idx.tz is None else idx.tz_convert("UTC")
        if daily_or_coarser:
            idx = idx.normalize()
        r = pd.Series(r.to_numpy(), index=idx, name=symbol)
        # Normalising can collide bars that were distinct instants; keep the
        # last, which is the one the session actually closed on.
        series[symbol] = r[~r.index.duplicated(keep="last")]
    return pd.concat(series, axis=1, sort=True)


class RiskEngine:
    def __init__(
        self,
        cfg: RiskConfig,
        universe: Universe | None = None,
        returns: pd.DataFrame | None = None,
        regimes: pd.Series | None = None,
        corr_window: int = 63,
        var_window_bars: int = 252,
        min_corr_obs: int = 20,
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
        min_corr_obs:
            Overlapping observations a pair needs before its correlation is
            believed. Below it the pair is unmeasured, not uncorrelated.
        """
        self._cfg = cfg
        self._universe = universe
        self._returns = returns
        self._regimes = regimes
        self._corr_window = corr_window
        self._var_window_bars = var_window_bars
        self._min_corr_obs = min_corr_obs
        self._pair_cache: dict[tuple[str, str, pd.Timestamp], float] = {}

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
        """Mean correlation to the current book, or ``nan`` if unmeasurable.

        The distinction between "uncorrelated" and "not measured" is the whole
        point of returning ``nan`` here. Both used to come back as 0.0, so a
        pair with no overlapping history was credited as a perfect diversifier
        and sized accordingly — the fail-open direction, and silent.

        ``min_periods`` is what makes that distinction possible. Without it a
        pair with one overlapping observation returns a correlation computed
        from a zero-variance slice, which is both meaningless and noisy enough
        to emit numpy divide-by-zero warnings from inside ``cov``.
        """
        if self._returns is None:
            # No return matrix was supplied: the penalty is not configured, as
            # opposed to configured and unmeasurable. Charging for it here
            # would silently halve every size for callers that never asked for
            # the feature.
            return 0.0
        if not holdings:
            return 0.0        # an empty book correlates with nothing; not a gap
        if symbol not in self._returns.columns:
            logger.warning(
                "%s: no return history in the risk matrix; sizing it as fully "
                "correlated with the book", symbol,
            )
            return float("nan")
        # The matrix index is normalised to UTC; `when` arrives from the panel
        # in whatever the vendor stamped. Match the index's tz-awareness or the
        # slice raises rather than returning the wrong window.
        when = _match_tz(when, self._returns.index)
        window = self._returns.loc[:when].tail(self._corr_window)
        if len(window) < self._corr_window // 2:
            return float("nan")
        cand = window[symbol]
        corrs = [
            c
            for h in holdings
            if h in window.columns
            and np.isfinite(c := cand.corr(window[h], min_periods=self._min_corr_obs))
        ]
        if not corrs:
            logger.warning(
                "%s: correlation to a %d-name book is unmeasurable (<%d overlapping "
                "observations); sizing it as fully correlated",
                symbol, len(holdings), self._min_corr_obs,
            )
            return float("nan")
        return float(np.mean(corrs))

    # ------------------------------------------------------------------ #

    def _pairwise(self, a: str, b: str, when: pd.Timestamp) -> float:
        """Correlation between two symbols, or ``nan`` when unmeasurable."""
        if self._returns is None:
            return float("nan")
        for s in (a, b):
            if s not in self._returns.columns:
                return float("nan")
        when = _match_tz(when, self._returns.index)
        key = (a, b, when)
        if key not in self._pair_cache:
            window = self._returns.loc[:when].tail(self._corr_window)
            if len(window) < self._corr_window // 2:
                self._pair_cache[key] = float("nan")
            else:
                self._pair_cache[key] = float(
                    window[a].corr(window[b], min_periods=self._min_corr_obs)
                )
        return self._pair_cache[key]

    def effective_heat(self, risks: Mapping[str, float], when: pd.Timestamp) -> float:
        """Portfolio risk-at-stop as ``sqrt(r' C r)`` rather than ``sum(r)``.

        The linear sum is the loss if every position stops out on the same bar.
        For a book that is one bet in five names that is a realistic Tuesday;
        for a genuinely diversified book it is a number with no probability
        attached to it. Summing treats the two identically, so it overstates
        the diversified book and — because the cap then binds on both at the
        same total — leaves the correlation penalty with nothing to do but
        redistribute size between names.

        The quadratic form fixes exactly that. At perfect correlation
        ``sqrt(r' C r)`` reduces to ``sum(r)``, so a concentrated book is
        sized as it always was. Below perfect correlation it is smaller, and
        the diversified book gets the room its structure has actually earned.

        Any pair whose correlation cannot be measured is taken as 1.0 — the
        conservative reading, and the one that degrades this back to the old
        linear behaviour rather than to something optimistic.
        """
        symbols = [s for s, r in risks.items() if r > 0]
        if not symbols:
            return 0.0
        r = np.array([risks[s] for s in symbols], dtype=float)
        if len(symbols) == 1:
            return float(r[0])
        total = 0.0
        for i, a in enumerate(symbols):
            total += r[i] * r[i]
            for j in range(i + 1, len(symbols)):
                rho = self._pairwise(a, symbols[j], when)
                if not np.isfinite(rho):
                    rho = 1.0
                total += 2.0 * rho * r[i] * r[j]
        return float(np.sqrt(max(total, 0.0)))

    def _heat_scale(
        self,
        plan: TradePlan,
        snapshot: PortfolioSnapshot,
        candidate_risk: float,
        heat_cap: float,
    ) -> float:
        """Largest fraction of the candidate that keeps effective heat capped.

        With ``A`` the book's current effective heat squared, ``x`` the cross
        term against the candidate and ``c`` the candidate's own risk, adding
        ``s`` of the candidate gives ``A + 2sx + s^2 c^2``. Solve that quadratic
        at the cap rather than searching: it is exact, and it degrades to the
        old linear ratio when everything is perfectly correlated.
        """
        held = {s: r for s, r in snapshot.symbol_risks.items() if r > 0}
        if not held:
            # No breakdown available (or an empty book): fall back to the sum,
            # which is what the caller's open_risk_fraction already is.
            available = heat_cap - snapshot.open_risk_fraction
            if available <= 0:
                return 0.0
            return min(available / candidate_risk, 1.0)

        when = plan.decision_date
        a = self.effective_heat(held, when) ** 2
        if a >= heat_cap**2:
            return 0.0
        cross = 0.0
        for sym, r in held.items():
            rho = self._pairwise(plan.symbol, sym, when)
            if not np.isfinite(rho):
                rho = 1.0
            cross += rho * r
        # c^2 s^2 + 2 (c * cross) s + (a - cap^2) <= 0
        qa = candidate_risk**2
        qb = 2.0 * candidate_risk * cross
        qc = a - heat_cap**2
        disc = qb * qb - 4.0 * qa * qc
        if disc <= 0 or qa <= 0:
            return 0.0
        s = (-qb + float(np.sqrt(disc))) / (2.0 * qa)
        return float(min(max(s, 0.0), 1.0))

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

        # Portfolio heat: total open risk stays under the cap, measured with
        # the book's correlation structure rather than by summing.
        entry_ref = plan.entry_ref if plan.entry_ref > 0 else plan.tp_price
        stop_frac = abs(entry_ref - plan.stop_price) / max(entry_ref, 1e-9)
        heat_cap = cfg.portfolio_heat_cap_pct / 100.0
        candidate_risk = size * stop_frac
        if candidate_risk <= 0:
            return 0.0
        scale = self._heat_scale(plan, snapshot, candidate_risk, heat_cap)
        if scale <= 0:
            return 0.0
        size *= scale

        # Sector concentration.
        if self._universe is not None:
            sector = self._universe.sector_of(plan.symbol)
            current = snapshot.sector_weights.get(sector, 0.0)
            room = cfg.max_sector_weight - current
            if room <= 0:
                return 0.0
            size = min(size, room)

        # Correlation penalty: up to 50% haircut above the threshold. An
        # unmeasurable correlation takes the full haircut rather than none —
        # the penalty is bounded at half the position, so the conservative
        # reading costs size, while the permissive one costs the entire point
        # of having the penalty.
        avg_corr = self._avg_correlation_to_book(
            plan.symbol, list(snapshot.symbol_weights), plan.decision_date
        )
        if not np.isfinite(avg_corr):
            avg_corr = 1.0
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
