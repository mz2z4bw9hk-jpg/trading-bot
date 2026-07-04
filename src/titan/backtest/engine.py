"""Event-driven portfolio backtester.

Execution conventions (deliberately conservative):

- Decisions made on bar ``t`` execute at the OPEN of ``t + lag`` (no same-bar
  fills, ever — that is how look-ahead sneaks into backtests).
- Slippage + commission are folded into the fill price via the cost model.
- Stops/take-profits are evaluated intrabar: gaps through a level fill at the
  open (you don't get your stop price in a gap), and when both stop and target
  are touched inside one bar the STOP is assumed to have been hit first
  (``stop_first_on_ambiguous_bar``).
- Portfolio-level constraints (position count, gross exposure) are enforced at
  entry; an optional :class:`RiskApprover` (the risk engine) can further scale
  or reject each candidate based on live portfolio state.

The engine never sees features, labels or probabilities — only trade plans.
Anything statistical must have happened strictly earlier in the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from titan.backtest.costs import CostModel
from titan.backtest.metrics import PerfSummary, summarize
from titan.core.config import BacktestConfig
from titan.core.log import get_logger
from titan.core.types import Side, Universe

logger = get_logger(__name__)


@dataclass(slots=True)
class TradePlan:
    """A fully specified trade intention (already signal- and risk-vetted)."""

    symbol: str
    decision_date: pd.Timestamp
    size_fraction: float  # target position notional as fraction of equity
    stop_price: float
    tp_price: float
    max_holding_bars: int
    side: Side = Side.LONG
    entry_ref: float = 0.0  # reference price at decision (close of decision bar)
    tag: str = ""

    def __post_init__(self) -> None:
        if self.side is Side.LONG and not (self.stop_price < self.tp_price):
            raise ValueError(f"{self.symbol}: long plan needs stop < tp")
        if self.size_fraction <= 0:
            raise ValueError("size_fraction must be positive")


@dataclass(slots=True)
class TradeRecord:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: float
    size_fraction: float
    pnl_cash: float
    pnl_fraction: float  # PnL / equity at entry
    bars_held: int
    exit_reason: str  # stop | tp | time | end
    mae: float  # max adverse excursion, fraction of entry price
    mfe: float  # max favorable excursion, fraction of entry price
    tag: str = ""

    def to_dict(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["entry_date"] = str(self.entry_date.date())
        d["exit_date"] = str(self.exit_date.date())
        for k in ("entry_price", "exit_price", "shares", "size_fraction",
                  "pnl_cash", "pnl_fraction", "mae", "mfe"):
            d[k] = round(float(d[k]), 6)
        return d


@dataclass(slots=True)
class _Position:
    symbol: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: float
    stop: float
    tp: float
    max_holding: int
    size_fraction: float
    entry_equity: float
    tag: str
    bars_held: int = 0
    low_water: float = np.inf
    high_water: float = -np.inf
    last_close: float = np.nan


@dataclass(slots=True)
class PortfolioSnapshot:
    """Live state handed to the risk approver at each entry decision."""

    equity: float
    n_positions: int
    gross_exposure: float
    open_risk_fraction: float  # sum of size * stop-distance across open positions
    symbol_weights: dict[str, float]
    sector_weights: dict[str, float]
    strategy_drawdown: float  # current drawdown of strategy equity


class RiskApprover(Protocol):
    def approve(self, plan: TradePlan, snapshot: PortfolioSnapshot) -> float:
        """Return the approved size fraction (0 rejects the plan)."""
        ...


@dataclass(slots=True)
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    exposure: pd.Series
    trades: list[TradeRecord]
    n_submitted: int
    n_rejected: int
    summary: PerfSummary

    def to_dict(self) -> dict:
        return {
            "summary": self.summary.to_dict(),
            "n_submitted": self.n_submitted,
            "n_rejected": self.n_rejected,
            "n_trades": len(self.trades),
        }


class BacktestEngine:
    def __init__(
        self,
        cfg: BacktestConfig,
        cost_model: CostModel | None = None,
        universe: Universe | None = None,
        risk_approver: RiskApprover | None = None,
    ) -> None:
        self._cfg = cfg
        self._costs = cost_model or CostModel(cfg.costs)
        self._universe = universe
        self._risk = risk_approver

    # ------------------------------------------------------------------ #

    @staticmethod
    def _daily_vol(frames: dict[str, pd.DataFrame], span: int = 21) -> dict[str, pd.Series]:
        out = {}
        for sym, df in frames.items():
            out[sym] = (
                np.log(df["close"]).diff().ewm(span=span, adjust=False, min_periods=5).std()
            ).fillna(0.01)
        return out

    def run(
        self,
        frames: dict[str, pd.DataFrame],
        plans: list[TradePlan],
        calendar: pd.DatetimeIndex | None = None,
        start: pd.Timestamp | None = None,
    ) -> BacktestResult:
        if calendar is None:
            all_dates: set[pd.Timestamp] = set()
            for df in frames.values():
                all_dates.update(df.index)
            calendar = pd.DatetimeIndex(sorted(all_dates))
        if start is not None:
            calendar = calendar[calendar >= start]
        if len(calendar) == 0:
            raise ValueError("empty calendar")
        pos_of_date = {ts: i for i, ts in enumerate(calendar)}
        vol = self._daily_vol(frames)

        # Schedule plans on execution dates (decision + lag on the calendar).
        scheduled: dict[pd.Timestamp, list[TradePlan]] = {}
        n_submitted = 0
        for plan in plans:
            n_submitted += 1
            p = pos_of_date.get(plan.decision_date)
            if p is None or p + self._cfg.execution_lag_bars >= len(calendar):
                continue
            exec_date = calendar[p + self._cfg.execution_lag_bars]
            scheduled.setdefault(exec_date, []).append(plan)

        cash = self._cfg.initial_capital
        peak_equity = cash
        positions: dict[str, _Position] = {}
        trades: list[TradeRecord] = []
        equity_curve: list[float] = []
        exposure_curve: list[float] = []
        n_rejected = 0

        def _mark_equity(ts: pd.Timestamp) -> float:
            value = cash
            for p in positions.values():
                frame = frames[p.symbol]
                if ts in frame.index:
                    p.last_close = float(frame.at[ts, "close"])
                value += p.shares * p.last_close
            return value

        def _close_position(p: _Position, ts: pd.Timestamp, raw_price: float, reason: str) -> float:
            nonlocal cash
            v = float(vol[p.symbol].get(ts, 0.01))
            fill = self._costs.apply_exit(raw_price, v, is_long=True)
            proceeds = p.shares * fill
            cash += proceeds
            pnl = proceeds - p.shares * p.entry_price
            trades.append(
                TradeRecord(
                    symbol=p.symbol,
                    entry_date=p.entry_date,
                    exit_date=ts,
                    entry_price=p.entry_price,
                    exit_price=fill,
                    shares=p.shares,
                    size_fraction=p.size_fraction,
                    pnl_cash=pnl,
                    pnl_fraction=pnl / p.entry_equity,
                    bars_held=p.bars_held,
                    exit_reason=reason,
                    mae=(p.low_water / p.entry_price) - 1.0 if np.isfinite(p.low_water) else 0.0,
                    mfe=(p.high_water / p.entry_price) - 1.0 if np.isfinite(p.high_water) else 0.0,
                    tag=p.tag,
                )
            )
            return pnl

        for ts in calendar:
            # ---- exits ---------------------------------------------------
            for sym in list(positions):
                p = positions[sym]
                frame = frames[sym]
                if ts not in frame.index:
                    continue
                if ts == p.entry_date:
                    continue  # entry bar handled at entry time
                bar = frame.loc[ts]
                p.bars_held += 1
                p.low_water = min(p.low_water, float(bar["low"]))
                p.high_water = max(p.high_water, float(bar["high"]))

                exit_price: float | None = None
                reason = ""
                o, h, low_ = float(bar["open"]), float(bar["high"]), float(bar["low"])
                if o <= p.stop:
                    exit_price, reason = o, "stop"  # gapped through the stop
                elif o >= p.tp:
                    exit_price, reason = o, "tp"
                else:
                    hit_stop = low_ <= p.stop
                    hit_tp = h >= p.tp
                    if hit_stop and hit_tp:
                        if self._cfg.stop_first_on_ambiguous_bar:
                            exit_price, reason = p.stop, "stop"
                        else:
                            exit_price, reason = p.tp, "tp"
                    elif hit_stop:
                        exit_price, reason = p.stop, "stop"
                    elif hit_tp:
                        exit_price, reason = p.tp, "tp"
                if exit_price is None and p.bars_held >= p.max_holding:
                    exit_price, reason = float(bar["close"]), "time"
                if exit_price is not None:
                    _close_position(p, ts, exit_price, reason)
                    del positions[sym]

            # ---- entries ---------------------------------------------------
            equity_now = _mark_equity(ts)
            peak_equity = max(peak_equity, equity_now)
            for plan in scheduled.get(ts, []):
                frame = frames.get(plan.symbol)
                if frame is None or ts not in frame.index or plan.symbol in positions:
                    n_rejected += 1
                    continue
                if len(positions) >= self._cfg.max_positions:
                    n_rejected += 1
                    continue
                gross = sum(
                    abs(p.shares * p.last_close) for p in positions.values()
                ) / max(equity_now, 1e-9)
                size = min(plan.size_fraction, self._cfg.max_gross_exposure - gross)
                if size <= 1e-6:
                    n_rejected += 1
                    continue
                if self._risk is not None:
                    snapshot = self._snapshot(positions, equity_now, peak_equity)
                    size = min(size, self._risk.approve(plan, snapshot))
                    if size <= 1e-6:
                        n_rejected += 1
                        continue
                bar = frame.loc[ts]
                v = float(vol[plan.symbol].get(ts, 0.01))
                fill = self._costs.apply_entry(float(bar["open"]), v, is_long=True)
                shares = size * equity_now / fill
                cash -= shares * fill
                positions[plan.symbol] = _Position(
                    symbol=plan.symbol,
                    entry_date=ts,
                    entry_price=fill,
                    shares=shares,
                    stop=plan.stop_price,
                    tp=plan.tp_price,
                    max_holding=plan.max_holding_bars,
                    size_fraction=size,
                    entry_equity=equity_now,
                    tag=plan.tag,
                    last_close=float(bar["close"]),
                    low_water=float(bar["low"]),
                    high_water=float(bar["high"]),
                )

            equity_now = _mark_equity(ts)
            peak_equity = max(peak_equity, equity_now)
            gross_value = sum(abs(p.shares * p.last_close) for p in positions.values())
            equity_curve.append(equity_now)
            exposure_curve.append(gross_value / max(equity_now, 1e-9))

        # ---- force-close at the end -----------------------------------
        last_ts = calendar[-1]
        for sym in list(positions):
            p = positions[sym]
            _close_position(p, last_ts, p.last_close, "end")
            del positions[sym]
        if equity_curve:
            equity_curve[-1] = cash

        equity = pd.Series(equity_curve, index=calendar, name="equity")
        returns = equity.pct_change().fillna(0.0)
        exposure = pd.Series(exposure_curve, index=calendar, name="exposure")
        result = BacktestResult(
            equity=equity,
            returns=returns,
            exposure=exposure,
            trades=trades,
            n_submitted=n_submitted,
            n_rejected=n_rejected,
            summary=summarize(equity, trades, exposure),
        )
        logger.info(
            "backtest: %d plans -> %d trades (%d rejected) | sharpe %.2f | maxDD %.1f%%",
            n_submitted, len(trades), n_rejected,
            result.summary.sharpe, 100 * result.summary.max_drawdown,
        )
        return result

    # ------------------------------------------------------------------ #

    def _snapshot(
        self, positions: dict[str, _Position], equity: float, peak_equity: float
    ) -> PortfolioSnapshot:
        weights = {
            sym: p.shares * p.last_close / max(equity, 1e-9) for sym, p in positions.items()
        }
        sectors: dict[str, float] = {}
        if self._universe is not None:
            for sym, w in weights.items():
                sector = self._universe.sector_of(sym)
                sectors[sector] = sectors.get(sector, 0.0) + w
        open_risk = sum(
            p.size_fraction * abs(p.entry_price - p.stop) / p.entry_price
            for p in positions.values()
        )
        return PortfolioSnapshot(
            equity=equity,
            n_positions=len(positions),
            gross_exposure=sum(abs(w) for w in weights.values()),
            open_risk_fraction=open_risk,
            symbol_weights=weights,
            sector_weights=sectors,
            strategy_drawdown=equity / max(peak_equity, 1e-9) - 1.0,
        )
