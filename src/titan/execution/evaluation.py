"""Execution evaluation: what we captured, and what it cost us to capture it.

The whole question for a liquidity provider fits in one identity:

    gross spread capture  =  realized spread  +  adverse selection

You earn the spread at the moment of the fill. Then the mid moves, and the
part of that move that goes against your new position is handed back. What
remains is realized spread — the only one of the three that is P&L.

That identity is why "we captured 1.8bps of spread" is not a result. A desk
capturing 1.8bps and paying 2.4bps of adverse selection is losing money while
every spread-capture dashboard it owns is green. The evaluator therefore never
reports capture without the markout beside it, and :meth:`EvaluationReport.
verdict` reads the sign of the *net*, not the capture.

Horizons are the diagnostic. Adverse selection at 1s is queue position and
latency; at 5 minutes it is whether the alpha was real. A book that looks
toxic at 1s and flat at 60s is being picked off by faster participants. One
that is flat at 1s and bleeding at 60s is quoting against genuine information.
Those have opposite fixes, so the report keeps every horizon separately rather
than collapsing to a single number.

Sign convention, stated once and used everywhere: all three quantities are
positive when they help the maker, EXCEPT ``adverse_selection_bps``, which is
positive when the mid moved against the position. So
``net = gross - adverse - fee``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from titan.agents.contracts import OrderSide
from titan.core.log import get_logger

logger = get_logger(__name__)

__all__ = [
    "EvaluationReport",
    "ExecutionEvaluator",
    "Fill",
    "FillEvaluation",
    "HorizonStats",
    "Markout",
    "MidTimeline",
]

DEFAULT_HORIZONS_S: tuple[float, ...] = (1.0, 5.0, 30.0, 60.0, 300.0)


@dataclass(frozen=True, slots=True)
class Fill:
    """One execution. ``fee_bps`` is negative for a maker rebate."""

    symbol: str
    ts: pd.Timestamp
    side: OrderSide
    price: float
    qty: float
    liquidity: str = "maker"          # "maker" | "taker"
    fee_bps: float = 0.0
    venue: str = ""
    order_id: str = ""
    verdict_id: str = ""              # ties the fill back to its risk decision

    def __post_init__(self) -> None:
        if self.price <= 0 or self.qty <= 0:
            raise ValueError(f"{self.symbol}: fill needs positive price and qty")
        if self.liquidity not in {"maker", "taker"}:
            raise ValueError(f"liquidity must be maker or taker, got {self.liquidity}")

    @property
    def notional(self) -> float:
        return self.price * self.qty


class MidTimeline:
    """As-of mid lookup for one symbol.

    Backed by a sorted series and ``asof`` semantics: the mid *at or before*
    the requested instant, never after. Interpolating toward a future quote
    would leak the very information the markout is trying to measure.
    """

    def __init__(self, mids: pd.Series) -> None:
        if not isinstance(mids.index, pd.DatetimeIndex):
            raise TypeError("MidTimeline requires a DatetimeIndex")
        s = mids.sort_index()
        self._mids = s[~s.index.duplicated(keep="last")].astype(float)
        if self._mids.empty:
            raise ValueError("MidTimeline requires at least one quote")

    @property
    def start(self) -> pd.Timestamp:
        return self._mids.index[0]

    @property
    def end(self) -> pd.Timestamp:
        return self._mids.index[-1]

    def at(self, ts: pd.Timestamp) -> float:
        """Mid at or before ``ts``; ``nan`` before the first quote."""
        if ts < self.start:
            return float("nan")
        idx = self._mids.index.asof(ts)
        return float("nan") if pd.isna(idx) else float(self._mids.loc[idx])

    def covers(self, ts: pd.Timestamp) -> bool:
        """Whether ``ts`` is inside the quoted window.

        A markout horizon reaching past the end of the data is not a zero
        markout — it is an unmeasured one, and must not be averaged in.
        """
        return self.start <= ts <= self.end


@dataclass(frozen=True, slots=True)
class Markout:
    """One fill measured at one horizon."""

    horizon_s: float
    mid_at_fill: float
    mid_at_horizon: float
    gross_capture_bps: float
    adverse_selection_bps: float
    realized_spread_bps: float
    net_bps: float
    measured: bool = True

    @property
    def toxicity_ratio(self) -> float:
        """Adverse selection as a multiple of gross capture.

        Above 1.0 the flow costs more than the spread pays. Undefined (``nan``)
        when nothing was captured, which is honest — a ratio against zero is
        not a large number, it is not a number.
        """
        if not math.isfinite(self.gross_capture_bps) or abs(self.gross_capture_bps) < 1e-12:
            return float("nan")
        return self.adverse_selection_bps / self.gross_capture_bps


@dataclass(frozen=True, slots=True)
class FillEvaluation:
    fill: Fill
    markouts: dict[float, Markout] = field(default_factory=dict)

    def at(self, horizon_s: float) -> Markout | None:
        return self.markouts.get(horizon_s)


@dataclass(frozen=True, slots=True)
class HorizonStats:
    """Aggregates at one horizon. ``n`` counts measured fills only."""

    horizon_s: float
    n: int
    notional: float
    gross_capture_bps: float
    adverse_selection_bps: float
    realized_spread_bps: float
    net_bps: float
    net_bps_median: float
    net_cash: float
    win_rate: float
    toxicity_ratio: float

    def to_dict(self) -> dict:
        return {
            "horizon_s": self.horizon_s,
            "n_fills": self.n,
            "notional": round(self.notional, 2),
            "gross_capture_bps": round(self.gross_capture_bps, 4),
            "adverse_selection_bps": round(self.adverse_selection_bps, 4),
            "realized_spread_bps": round(self.realized_spread_bps, 4),
            "net_bps": round(self.net_bps, 4),
            "net_bps_median": round(self.net_bps_median, 4),
            "net_cash": round(self.net_cash, 2),
            "win_rate": round(self.win_rate, 4),
            "toxicity_ratio": (
                None if not math.isfinite(self.toxicity_ratio)
                else round(self.toxicity_ratio, 4)
            ),
        }


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    horizons: tuple[float, ...]
    stats: dict[float, HorizonStats]
    by_bucket: dict[str, dict[str, HorizonStats]]
    n_fills: int
    n_unmeasured: int
    decision_horizon_s: float

    @property
    def decision(self) -> HorizonStats:
        return self.stats[self.decision_horizon_s]

    def verdict(self) -> str:
        """One line a human can act on, read off the net rather than the capture."""
        d = self.decision
        if d.n == 0:
            return "NO DATA: no fill had a measurable markout at the decision horizon"
        if d.net_bps <= 0:
            return (
                f"UNPROFITABLE: {d.net_bps:+.2f}bps net at {d.horizon_s:g}s — "
                f"captured {d.gross_capture_bps:.2f}, paid "
                f"{d.adverse_selection_bps:.2f} adverse selection"
            )
        if d.toxicity_ratio > 0.7:
            return (
                f"THIN: {d.net_bps:+.2f}bps net, but adverse selection is "
                f"{d.toxicity_ratio:.0%} of capture — little margin for a "
                "worse fill rate or a wider queue"
            )
        return (
            f"HEALTHY: {d.net_bps:+.2f}bps net at {d.horizon_s:g}s, adverse "
            f"selection {d.toxicity_ratio:.0%} of capture"
        )

    def to_dict(self) -> dict:
        return {
            "n_fills": self.n_fills,
            "n_unmeasured": self.n_unmeasured,
            "decision_horizon_s": self.decision_horizon_s,
            "verdict": self.verdict(),
            "horizons": {str(h): s.to_dict() for h, s in self.stats.items()},
            "by_bucket": {
                dim: {k: v.to_dict() for k, v in buckets.items()}
                for dim, buckets in self.by_bucket.items()
            },
        }


class ExecutionEvaluator:
    """Scores fills against the mid that followed them.

    Stateless per call — hand it fills and quotes, get a report. That makes it
    usable identically over a shadow-mode session, a backtest, and yesterday's
    live tape, which is the point: an execution metric computed one way in
    research and another way in production is a metric nobody trusts.
    """

    def __init__(
        self,
        timelines: dict[str, MidTimeline],
        horizons: Sequence[float] = DEFAULT_HORIZONS_S,
        *,
        decision_horizon_s: float | None = None,
    ) -> None:
        if not horizons:
            raise ValueError("at least one horizon is required")
        self._timelines = timelines
        self._horizons = tuple(sorted(float(h) for h in horizons))
        chosen = decision_horizon_s if decision_horizon_s is not None else self._horizons[-1]
        if chosen not in self._horizons:
            raise ValueError(f"decision horizon {chosen} is not among {self._horizons}")
        self._decision = chosen

    # ------------------------------------------------------------------ #

    def evaluate_fill(self, fill: Fill) -> FillEvaluation:
        """Decompose one fill at every horizon.

        ``position_sign`` is the direction of the exposure the fill creates:
        +1 after a buy, -1 after a sell. Gross capture is measured against the
        mid at the fill; adverse selection is the subsequent move weighted by
        that sign, so it is positive whenever the market moved the wrong way.
        """
        timeline = self._timelines.get(fill.symbol)
        if timeline is None:
            logger.warning("no mid timeline for %s; fill unmeasured", fill.symbol)
            return FillEvaluation(fill=fill, markouts={})

        m0 = timeline.at(fill.ts)
        if not math.isfinite(m0) or m0 <= 0:
            return FillEvaluation(fill=fill, markouts={})

        sign = fill.side.sign
        # Positive when the fill happened on the maker's favourable side of the
        # mid: bought below it, or sold above it.
        gross = 1e4 * sign * (m0 - fill.price) / m0

        markouts: dict[float, Markout] = {}
        for h in self._horizons:
            future_ts = fill.ts + pd.Timedelta(seconds=h)
            if not timeline.covers(future_ts):
                markouts[h] = Markout(
                    horizon_s=h, mid_at_fill=m0, mid_at_horizon=float("nan"),
                    gross_capture_bps=gross, adverse_selection_bps=float("nan"),
                    realized_spread_bps=float("nan"), net_bps=float("nan"),
                    measured=False,
                )
                continue
            m1 = timeline.at(future_ts)
            if not math.isfinite(m1) or m1 <= 0:
                markouts[h] = Markout(
                    horizon_s=h, mid_at_fill=m0, mid_at_horizon=float("nan"),
                    gross_capture_bps=gross, adverse_selection_bps=float("nan"),
                    realized_spread_bps=float("nan"), net_bps=float("nan"),
                    measured=False,
                )
                continue

            adverse = -1e4 * sign * (m1 - m0) / m0
            realized = gross - adverse
            markouts[h] = Markout(
                horizon_s=h,
                mid_at_fill=m0,
                mid_at_horizon=m1,
                gross_capture_bps=gross,
                adverse_selection_bps=adverse,
                realized_spread_bps=realized,
                net_bps=realized - fill.fee_bps,
                measured=True,
            )
        return FillEvaluation(fill=fill, markouts=markouts)

    # ------------------------------------------------------------------ #

    def run(
        self,
        fills: Iterable[Fill],
        *,
        buckets: Sequence[str] = ("venue", "liquidity", "symbol"),
    ) -> EvaluationReport:
        """Evaluate every fill and aggregate, overall and per bucket."""
        evaluations = [self.evaluate_fill(f) for f in fills]
        if not evaluations:
            raise ValueError("no fills to evaluate")

        stats = {
            h: self._aggregate(evaluations, h) for h in self._horizons
        }
        by_bucket: dict[str, dict[str, HorizonStats]] = {}
        for dim in buckets:
            groups: dict[str, list[FillEvaluation]] = {}
            for ev in evaluations:
                groups.setdefault(str(getattr(ev.fill, dim)), []).append(ev)
            by_bucket[dim] = {
                key: self._aggregate(group, self._decision)
                for key, group in sorted(groups.items())
            }

        n_unmeasured = sum(
            1 for ev in evaluations
            if (m := ev.at(self._decision)) is None or not m.measured
        )
        report = EvaluationReport(
            horizons=self._horizons,
            stats=stats,
            by_bucket=by_bucket,
            n_fills=len(evaluations),
            n_unmeasured=n_unmeasured,
            decision_horizon_s=self._decision,
        )
        logger.info(
            "execution eval: %d fills (%d unmeasured at %gs) -> %s",
            report.n_fills, report.n_unmeasured, self._decision, report.verdict(),
        )
        return report

    # ------------------------------------------------------------------ #

    @staticmethod
    def _aggregate(evaluations: Sequence[FillEvaluation], horizon: float) -> HorizonStats:
        """Notional-weight the bps figures; a 10x fill is 10x the evidence.

        An equal-weighted mean of per-fill bps is the classic way to report a
        profitable desk that is losing money: the losses arrive in size and the
        gains in odd lots, and the unweighted average never sees it.
        """
        rows = [
            (ev.fill, m) for ev in evaluations
            if (m := ev.at(horizon)) is not None and m.measured
        ]
        if not rows:
            return HorizonStats(
                horizon_s=horizon, n=0, notional=0.0, gross_capture_bps=float("nan"),
                adverse_selection_bps=float("nan"), realized_spread_bps=float("nan"),
                net_bps=float("nan"), net_bps_median=float("nan"), net_cash=0.0,
                win_rate=float("nan"), toxicity_ratio=float("nan"),
            )

        w = np.array([f.notional for f, _ in rows], dtype=float)
        total = float(w.sum())
        gross = np.array([m.gross_capture_bps for _, m in rows], dtype=float)
        adverse = np.array([m.adverse_selection_bps for _, m in rows], dtype=float)
        realized = np.array([m.realized_spread_bps for _, m in rows], dtype=float)
        net = np.array([m.net_bps for _, m in rows], dtype=float)

        def wmean(x: np.ndarray) -> float:
            return float(np.average(x, weights=w)) if total > 0 else float(np.mean(x))

        gross_m, adverse_m = wmean(gross), wmean(adverse)
        return HorizonStats(
            horizon_s=horizon,
            n=len(rows),
            notional=total,
            gross_capture_bps=gross_m,
            adverse_selection_bps=adverse_m,
            realized_spread_bps=wmean(realized),
            net_bps=wmean(net),
            net_bps_median=float(np.median(net)),
            net_cash=float(np.sum(net * w) / 1e4),
            win_rate=float(np.mean(net > 0)),
            toxicity_ratio=(
                adverse_m / gross_m if abs(gross_m) > 1e-12 else float("nan")
            ),
        )
