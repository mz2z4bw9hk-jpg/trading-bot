"""The typed boundary between the three agents.

Separation of concerns is worth nothing if it is a naming convention. These
types make the pipeline's one safety property structural:

    Signal -> Intent -> [Risk] -> ClearedOrder -> Execution

An :class:`Intent` carries no size and no price. That is the load-bearing
decision in this module. A signal agent that could size its own orders would
make the risk agent advisory — it could only ever shrink what it was handed,
never be the thing that decided. Sizing is a portfolio question (what else do
we hold, how much can we lose, how toxic is the flow) and the signal agent
knows none of that by construction, because none of it is on its inputs.

A :class:`ClearedOrder` is the only thing the execution agent accepts, and it
cannot be built without the private mint token in this module. Python cannot
truly seal that, and pretending otherwise would be theatre; what it gives us
is a single greppable construction site, which is what an auditor actually
wants. ``tests/test_agent_boundary.py`` fails the build if any module other
than the risk agent references the token.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd

__all__ = [
    "CheckResult",
    "ClearedOrder",
    "Disposition",
    "Intent",
    "OrderSide",
    "RiskBypassError",
    "RiskVerdict",
]


class RiskBypassError(RuntimeError):
    """Raised when an order reaches execution without passing risk."""


class OrderSide(StrEnum):
    """The direction of an order, distinct from a position's side.

    ``titan.core.types.Side`` describes what a position IS (long/short/flat).
    This describes what an order DOES. They coincide when opening and disagree
    when closing, and collapsing them is how a close gets sized as an open.
    """

    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is OrderSide.BUY else -1

    @property
    def opposite(self) -> OrderSide:
        return OrderSide.SELL if self is OrderSide.BUY else OrderSide.BUY


@dataclass(frozen=True, slots=True)
class Intent:
    """What the signal agent wants to do. Deliberately not an order.

    ``edge_bps`` is the expected gross edge over ``horizon_s``, before costs
    and before adverse selection. The risk agent subtracts both. A signal
    agent reporting an edge net of costs it cannot see would be reporting a
    number nobody can check.
    """

    symbol: str
    side: OrderSide
    edge_bps: float
    horizon_s: float
    confidence: float
    ts: pd.Timestamp
    source: str = ""
    features: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")
        if self.horizon_s <= 0:
            raise ValueError(f"horizon_s must be positive, got {self.horizon_s}")


class Disposition(StrEnum):
    APPROVED = "approved"      # cleared at the size risk chose
    RESIZED = "resized"        # cleared, but a constraint bound
    REJECTED = "rejected"      # not cleared


@dataclass(frozen=True, slots=True)
class CheckResult:
    """One risk check's outcome.

    ``scale`` is a multiplicative cap in [0, 1] applied to the base size. A
    check either rejects outright (``passed=False``) or expresses its opinion
    as a scale — there is no third option, which keeps the composition rule
    trivial and total.
    """

    name: str
    passed: bool
    scale: float = 1.0
    detail: str = ""

    def __post_init__(self) -> None:
        if not 0.0 <= self.scale <= 1.0:
            raise ValueError(f"{self.name}: scale must be in [0, 1], got {self.scale}")


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """The full audit record of one risk decision.

    Every check that ran is retained, passed or not, because the interesting
    question after a bad day is never "what rejected this" but "what nearly
    did". ``binding_constraint`` names the check that actually set the size.
    """

    intent: Intent
    disposition: Disposition
    approved_qty: float
    binding_constraint: str
    detail: str
    checks: tuple[CheckResult, ...] = ()
    verdict_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    @property
    def is_cleared(self) -> bool:
        return self.disposition is not Disposition.REJECTED and self.approved_qty > 0

    def to_dict(self) -> dict:
        return {
            "verdict_id": self.verdict_id,
            "ts": str(self.intent.ts),
            "symbol": self.intent.symbol,
            "side": str(self.intent.side),
            "source": self.intent.source,
            "edge_bps": round(self.intent.edge_bps, 4),
            "disposition": str(self.disposition),
            "approved_qty": round(self.approved_qty, 8),
            "binding_constraint": self.binding_constraint,
            "detail": self.detail,
            "checks": [
                {
                    "name": c.name,
                    "passed": c.passed,
                    "scale": round(c.scale, 6),
                    "detail": c.detail,
                }
                for c in self.checks
            ],
        }


# The mint. Private by convention, enforced by test_agent_boundary.py, which
# fails if anything outside the risk agent imports it.
_MINT = object()


@dataclass(frozen=True, slots=True)
class ClearedOrder:
    """An intent that has passed risk, and the only input execution accepts."""

    mint: object
    intent: Intent
    qty: float
    limit_price: float | None
    verdict_id: str

    def __post_init__(self) -> None:
        if self.mint is not _MINT:
            raise RiskBypassError(
                f"{self.intent.symbol}: ClearedOrder built without the risk mint. "
                "Orders reach execution through RiskAgent.clear() or not at all."
            )
        if self.qty <= 0:
            raise ValueError(f"{self.intent.symbol}: cleared qty must be positive")

    @property
    def symbol(self) -> str:
        return self.intent.symbol

    @property
    def side(self) -> OrderSide:
        return self.intent.side
