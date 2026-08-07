"""Risk & Monitoring Agent: the only path from an intent to an order.

The agent runs an ordered battery of checks over (intent, book, toxicity,
portfolio state) and returns a :class:`RiskVerdict`. Four properties matter
more than the individual limits, and each is a design decision rather than an
implementation detail:

**Fail closed.** Anything the agent cannot evaluate is a rejection. A missing
book, an unmeasurable VPIN, an exception inside a check — all reject. The
opposite convention (unknown means fine) fails precisely when the measurement
breaks, which is correlated with when the market is doing something unusual.

**Risk sizes, not signal.** The intent has no quantity. The base size comes
from the agent's own budget model and is then capped by every check. There is
no size on the input to be overridden, so there is no path where a signal
talks risk into a bigger position.

**Hard limits reject; soft limits scale.** A breach of a position or loss
limit is not negotiable and returns zero. Everything else expresses itself as
a multiplicative scale in [0, 1], and the smallest one binds. Composition is
therefore a single ``min``, and the binding constraint is always identifiable
by name — which is the number you need at 4pm, not a total.

**Every decision is recorded whole.** The verdict keeps all check results,
including the ones that passed, and is written to the audit log before the
order can be cleared.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from titan.agents.contracts import (
    _MINT,
    CheckResult,
    ClearedOrder,
    Disposition,
    Intent,
    OrderSide,
    RiskVerdict,
)
from titan.core.log import get_logger
from titan.microstructure.book import BookSnapshot
from titan.microstructure.toxicity import ToxicityState

logger = get_logger(__name__)

__all__ = ["PortfolioState", "RiskAgent", "RiskLimits"]


class RiskLimits(BaseModel):
    """Hard constraints. Every one of these is a number a human signed off on."""

    # -- capital ------------------------------------------------------- #
    equity: float = Field(1_000_000.0, gt=0)
    max_order_notional: float = Field(50_000.0, gt=0)
    max_position_notional: float = Field(150_000.0, gt=0)
    max_gross_notional: float = Field(1_000_000.0, gt=0)
    max_net_notional: float = Field(400_000.0, gt=0)

    # -- loss ---------------------------------------------------------- #
    kill_drawdown_pct: float = Field(4.0, gt=0)
    """Session drawdown, in percent of starting equity, that halts new risk."""
    warn_drawdown_pct: float = Field(2.0, gt=0)
    """Where sizing begins to taper toward the kill level."""

    # -- market quality ------------------------------------------------ #
    max_quote_age_ms: float = Field(500.0, gt=0)
    max_spread_bps: float = Field(25.0, gt=0)
    min_touch_size: float = Field(0.0, ge=0)

    # -- toxicity ------------------------------------------------------ #
    max_vpin: float = Field(0.55, ge=0, le=1)
    max_adverse_ofi: float = Field(2.0, ge=0)
    """Normalised OFI *against* the intent that is tolerated before rejecting."""
    min_edge_after_toxicity_bps: float = Field(0.5)
    """Edge must survive expected adverse selection by this margin."""

    # -- participation -------------------------------------------------- #
    max_touch_participation: float = Field(0.25, gt=0, le=1)
    """Order size as a fraction of the size resting on the far touch."""

    # -- sizing --------------------------------------------------------- #
    base_risk_fraction: float = Field(0.002, gt=0)
    """Fraction of equity the agent risks on a full-confidence intent."""
    min_order_notional: float = Field(500.0, gt=0)
    """Below this an order is dust: the fee and the queue cost exceed the edge."""


@dataclass(slots=True)
class PortfolioState:
    """Live book state the agent reasons against. Supplied, never inferred."""

    positions: dict[str, float] = field(default_factory=dict)   # symbol -> signed qty
    marks: dict[str, float] = field(default_factory=dict)       # symbol -> mark price
    session_pnl: float = 0.0
    session_high_water: float = 0.0
    halted: frozenset[str] = frozenset()

    def notional(self, symbol: str) -> float:
        return abs(self.positions.get(symbol, 0.0)) * self.marks.get(symbol, 0.0)

    def signed_notional(self, symbol: str) -> float:
        return self.positions.get(symbol, 0.0) * self.marks.get(symbol, 0.0)

    @property
    def gross_notional(self) -> float:
        return sum(abs(q) * self.marks.get(s, 0.0) for s, q in self.positions.items())

    @property
    def net_notional(self) -> float:
        return sum(q * self.marks.get(s, 0.0) for s, q in self.positions.items())

    def drawdown_pct(self, equity: float) -> float:
        """Session drawdown from the high-water mark, in percent of equity."""
        if equity <= 0:
            return 0.0
        peak = max(self.session_high_water, self.session_pnl, 0.0)
        return 100.0 * max(peak - self.session_pnl, 0.0) / equity


# Signature every check shares.
Check = Callable[
    ["RiskAgent", Intent, BookSnapshot, ToxicityState, PortfolioState, float],
    CheckResult,
]


class RiskAgent:
    """Intercepts every intent. Nothing reaches execution around it."""

    def __init__(self, limits: RiskLimits | None = None) -> None:
        self._limits = limits or RiskLimits()
        self._audit: list[RiskVerdict] = []

    @property
    def limits(self) -> RiskLimits:
        return self._limits

    @property
    def audit_log(self) -> tuple[RiskVerdict, ...]:
        return tuple(self._audit)

    # ------------------------------------------------------------------ #
    # Checks. Order matters: cheap and absolute first, so an obviously dead
    # intent never reaches an estimator that costs anything to evaluate.
    # ------------------------------------------------------------------ #

    def _check_kill_switch(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        if intent.symbol in state.halted:
            return CheckResult("kill_switch", False, 0.0, f"{intent.symbol} is halted")
        dd = state.drawdown_pct(self._limits.equity)
        if dd >= self._limits.kill_drawdown_pct:
            return CheckResult(
                "kill_switch", False, 0.0,
                f"session drawdown {dd:.2f}% >= kill {self._limits.kill_drawdown_pct}%",
            )
        # Taper between warn and kill rather than trading full size up to a
        # cliff. A book that is losing is a book whose edge estimate is
        # currently being contradicted by the market.
        warn = self._limits.warn_drawdown_pct
        if dd > warn:
            span = max(self._limits.kill_drawdown_pct - warn, 1e-9)
            scale = max(0.0, 1.0 - (dd - warn) / span)
            return CheckResult(
                "kill_switch", True, scale, f"drawdown {dd:.2f}% tapering size"
            )
        return CheckResult("kill_switch", True, 1.0, f"drawdown {dd:.2f}%")

    def _check_book_integrity(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        if book.symbol != intent.symbol:
            return CheckResult(
                "book_integrity", False, 0.0,
                f"book is for {book.symbol}, intent for {intent.symbol}",
            )
        if book.is_empty:
            return CheckResult("book_integrity", False, 0.0, "one-sided or empty book")
        if book.is_crossed:
            return CheckResult(
                "book_integrity", False, 0.0,
                f"crossed book: bid {book.best_bid} > ask {book.best_ask}",
            )
        if book.is_locked:
            return CheckResult("book_integrity", False, 0.0, "locked book")
        if not book.is_ordered:
            return CheckResult("book_integrity", False, 0.0, "levels not monotone")
        age = book.age_ms(intent.ts)
        if age < 0:
            return CheckResult(
                "book_integrity", False, 0.0, f"book postdates intent by {-age:.0f}ms"
            )
        if age > self._limits.max_quote_age_ms:
            return CheckResult(
                "book_integrity", False, 0.0,
                f"quote {age:.0f}ms old > {self._limits.max_quote_age_ms:.0f}ms",
            )
        if book.spread_bps > self._limits.max_spread_bps:
            return CheckResult(
                "book_integrity", False, 0.0,
                f"spread {book.spread_bps:.1f}bps > {self._limits.max_spread_bps}bps",
            )
        qb, qa = book.depth(1)
        if min(qb, qa) < self._limits.min_touch_size:
            return CheckResult(
                "book_integrity", False, 0.0,
                f"touch size {min(qb, qa):.4g} below floor",
            )
        return CheckResult("book_integrity", True, 1.0, f"spread {book.spread_bps:.1f}bps")

    def _check_toxicity(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        """Unmeasurable toxicity rejects. This is the fail-closed rule's teeth."""
        if not math.isfinite(tox.vpin):
            return CheckResult("toxicity", False, 0.0, "VPIN unmeasurable")
        if tox.vpin > self._limits.max_vpin:
            return CheckResult(
                "toxicity", False, 0.0,
                f"VPIN {tox.vpin:.3f} > {self._limits.max_vpin}",
            )
        # OFI pushing against the side we want to take means the queue ahead of
        # us is being consumed by flow that disagrees with us.
        if math.isfinite(tox.ofi):
            adverse = -intent.side.sign * tox.ofi
            if adverse > self._limits.max_adverse_ofi:
                return CheckResult(
                    "toxicity", False, 0.0,
                    f"OFI {adverse:.2f} against a {intent.side} intent",
                )
        # Taper across the TOP HALF of the tolerated range only. A haircut that
        # begins at VPIN zero charges every order for toxicity that is merely
        # normal, and — worse — means no order is ever APPROVED rather than
        # RESIZED, so the disposition stops distinguishing a routine fill from
        # one a limit actually bound.
        onset = 0.5 * self._limits.max_vpin
        if tox.vpin <= onset:
            return CheckResult("toxicity", True, 1.0, f"VPIN {tox.vpin:.3f} benign")
        span = max(self._limits.max_vpin - onset, 1e-9)
        scale = max(0.05, 1.0 - (tox.vpin - onset) / span)
        return CheckResult(
            "toxicity", True, scale, f"VPIN {tox.vpin:.3f} elevated, tapering"
        )

    def _check_edge_survives_costs(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        """The intent's edge must clear the spread it crosses and the impact it causes."""
        adverse = tox.expected_adverse_bps(qty, book.mid)
        if not math.isfinite(adverse):
            return CheckResult("edge_vs_costs", False, 0.0, "price impact unmeasurable")
        half_spread = 0.5 * book.spread_bps
        net = intent.edge_bps - adverse - half_spread
        if net < self._limits.min_edge_after_toxicity_bps:
            return CheckResult(
                "edge_vs_costs", False, 0.0,
                f"edge {intent.edge_bps:.2f}bps - adverse {adverse:.2f} - "
                f"half-spread {half_spread:.2f} = {net:.2f}bps, below floor "
                f"{self._limits.min_edge_after_toxicity_bps}bps",
            )
        return CheckResult("edge_vs_costs", True, 1.0, f"net edge {net:.2f}bps")

    def _check_position_limits(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        """Per-symbol, gross and net ceilings, expressed as room remaining."""
        lim = self._limits
        mid = book.mid
        want = qty * mid
        if want <= 0:
            return CheckResult("position_limits", False, 0.0, "non-positive notional")

        signed = state.signed_notional(intent.symbol)
        projected = abs(signed + intent.side.sign * want)
        # Reducing an existing position is not a new risk-taking act, so a
        # symbol already at its cap can still be traded out of.
        reducing = signed != 0.0 and (signed > 0) != (intent.side is OrderSide.BUY)

        room = []
        if not reducing:
            per_symbol = lim.max_position_notional - abs(signed)
            room.append(("max_position_notional", per_symbol))
            room.append(("max_gross_notional", lim.max_gross_notional - state.gross_notional))
            net_after = state.net_notional + intent.side.sign * want
            if abs(net_after) > abs(state.net_notional):
                room.append(
                    ("max_net_notional", lim.max_net_notional - abs(state.net_notional))
                )
        room.append(("max_order_notional", lim.max_order_notional))

        name, allowed = min(room, key=lambda kv: kv[1])
        if allowed <= 0:
            return CheckResult(
                "position_limits", False, 0.0,
                f"{name} exhausted (projected {projected:,.0f})",
            )
        scale = min(allowed / want, 1.0)
        detail = f"{name} allows {allowed:,.0f} of {want:,.0f}"
        return CheckResult("position_limits", True, scale, detail)

    def _check_participation(
        self, intent: Intent, book: BookSnapshot, tox: ToxicityState,
        state: PortfolioState, qty: float,
    ) -> CheckResult:
        """Do not be more than a fraction of the size we would trade against."""
        far_side = book.asks if intent.side is OrderSide.BUY else book.bids
        available = far_side[0].size
        if available <= 0:
            return CheckResult("participation", False, 0.0, "no size at the far touch")
        cap = self._limits.max_touch_participation * available
        scale = min(cap / qty, 1.0) if qty > 0 else 0.0
        return CheckResult(
            "participation", True, scale,
            f"{qty:.4g} vs {available:.4g} at touch (cap {cap:.4g})",
        )

    _CHECKS: tuple[str, ...] = (
        "_check_kill_switch",
        "_check_book_integrity",
        "_check_toxicity",
        "_check_edge_survives_costs",
        "_check_position_limits",
        "_check_participation",
    )

    # ------------------------------------------------------------------ #

    def _base_qty(self, intent: Intent, book: BookSnapshot) -> float:
        """Risk's own opinion of size, before any check has spoken.

        Confidence-scaled fraction of equity. The intent contributes its
        confidence and nothing else — it does not get to propose a notional.
        """
        budget = self._limits.equity * self._limits.base_risk_fraction
        return max(budget * intent.confidence, 0.0) / max(book.mid, 1e-12)

    def evaluate(
        self,
        intent: Intent,
        book: BookSnapshot,
        toxicity: ToxicityState,
        state: PortfolioState,
    ) -> RiskVerdict:
        """Run every check and return the verdict. Never raises."""
        results: list[CheckResult] = []
        try:
            qty = self._base_qty(intent, book)
            for name in self._CHECKS:
                check = getattr(self, name)
                result = check(intent, book, toxicity, state, qty)
                results.append(result)
                if not result.passed:
                    return self._record(
                        RiskVerdict(
                            intent=intent,
                            disposition=Disposition.REJECTED,
                            approved_qty=0.0,
                            binding_constraint=result.name,
                            detail=result.detail,
                            checks=tuple(results),
                        )
                    )
        except Exception as exc:  # fail closed, loudly
            logger.exception("risk check raised for %s; rejecting", intent.symbol)
            return self._record(
                RiskVerdict(
                    intent=intent,
                    disposition=Disposition.REJECTED,
                    approved_qty=0.0,
                    binding_constraint="internal_error",
                    detail=f"{type(exc).__name__}: {exc}",
                    checks=tuple(results),
                )
            )

        binding = min(results, key=lambda c: c.scale)
        approved = qty * binding.scale
        notional = approved * book.mid
        if notional < self._limits.min_order_notional:
            return self._record(
                RiskVerdict(
                    intent=intent,
                    disposition=Disposition.REJECTED,
                    approved_qty=0.0,
                    binding_constraint="min_order_notional",
                    detail=(
                        f"{notional:,.0f} after {binding.name} is below the "
                        f"{self._limits.min_order_notional:,.0f} floor"
                    ),
                    checks=tuple(results),
                )
            )

        disposition = (
            Disposition.APPROVED if binding.scale >= 0.999 else Disposition.RESIZED
        )
        return self._record(
            RiskVerdict(
                intent=intent,
                disposition=disposition,
                approved_qty=approved,
                binding_constraint=binding.name,
                detail=binding.detail,
                checks=tuple(results),
            )
        )

    def clear(self, verdict: RiskVerdict, limit_price: float | None = None) -> ClearedOrder:
        """Mint the execution agent's input. The only call site of the mint.

        Takes a verdict this agent produced rather than an intent, so an order
        cannot be cleared without the audit record that justifies it existing
        first and being retrievable by ``verdict_id``.
        """
        if not verdict.is_cleared:
            raise ValueError(
                f"{verdict.intent.symbol}: verdict {verdict.verdict_id} is "
                f"{verdict.disposition} ({verdict.binding_constraint}); nothing to clear"
            )
        if not any(v.verdict_id == verdict.verdict_id for v in self._audit):
            raise ValueError(
                f"verdict {verdict.verdict_id} is not in this agent's audit log"
            )
        return ClearedOrder(
            mint=_MINT,
            intent=verdict.intent,
            qty=verdict.approved_qty,
            limit_price=limit_price,
            verdict_id=verdict.verdict_id,
        )

    # ------------------------------------------------------------------ #

    def _record(self, verdict: RiskVerdict) -> RiskVerdict:
        self._audit.append(verdict)
        log = logger.info if verdict.is_cleared else logger.warning
        log(
            "risk %s %s %s: %s (%s) qty=%.6g",
            verdict.verdict_id, verdict.disposition, verdict.intent.symbol,
            verdict.binding_constraint, verdict.detail, verdict.approved_qty,
        )
        return verdict
