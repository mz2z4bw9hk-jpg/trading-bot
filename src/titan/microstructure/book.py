"""Limit order book snapshots and the state derived from one.

Everything here is a pure function of a single snapshot. Anything requiring
two snapshots (order flow imbalance) or a trade tape (VPIN, Kyle's lambda)
lives in :mod:`titan.microstructure.toxicity`, because those need a history
and therefore need to say what they do when the history is short.

The book is the primitive the signal and risk agents share. It is frozen: a
snapshot is an observation, and an observation that can be mutated after the
fact is not evidence of anything.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class BookLevel:
    price: float
    size: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.price) or self.price <= 0:
            raise ValueError(f"level price must be positive and finite, got {self.price}")
        if not math.isfinite(self.size) or self.size < 0:
            raise ValueError(f"level size must be non-negative and finite, got {self.size}")


@dataclass(frozen=True, slots=True)
class BookSnapshot:
    """One instrument's visible book at one instant.

    ``bids`` descend in price and ``asks`` ascend — index 0 is the touch on
    both sides. The constructor does not sort for you: a feed that delivers
    levels out of order is a feed bug, and silently repairing it here would
    hide it from the risk agent's integrity check, which is the one place that
    is supposed to notice.
    """

    symbol: str
    ts: pd.Timestamp
    bids: tuple[BookLevel, ...]
    asks: tuple[BookLevel, ...]

    # -- validity ------------------------------------------------------- #

    @property
    def is_empty(self) -> bool:
        return not self.bids or not self.asks

    @property
    def is_crossed(self) -> bool:
        """Best bid strictly above best ask — never real, always a feed fault."""
        return not self.is_empty and self.bids[0].price > self.asks[0].price

    @property
    def is_locked(self) -> bool:
        """Best bid equal to best ask. Legal on some venues, quotable on none."""
        return not self.is_empty and self.bids[0].price == self.asks[0].price

    @property
    def is_ordered(self) -> bool:
        """Levels monotone away from the touch, as the feed contract requires."""
        bid_ok = all(a.price > b.price for a, b in zip(self.bids, self.bids[1:]))
        ask_ok = all(a.price < b.price for a, b in zip(self.asks, self.asks[1:]))
        return bid_ok and ask_ok

    # -- prices --------------------------------------------------------- #

    @property
    def best_bid(self) -> float:
        return self.bids[0].price

    @property
    def best_ask(self) -> float:
        return self.asks[0].price

    @property
    def mid(self) -> float:
        return 0.5 * (self.best_bid + self.best_ask)

    @property
    def spread(self) -> float:
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> float:
        return 1e4 * self.spread / self.mid

    @property
    def microprice(self) -> float:
        """Size-weighted fair value: the mid leans toward the thinner side.

        Weighting is deliberately crossed — bid size multiplies the ASK price.
        Depth on the bid is buying pressure, so it pulls fair value up, and the
        naive same-side weighting gets the sign of that exactly backwards.
        """
        qb, qa = self.bids[0].size, self.asks[0].size
        total = qb + qa
        if total <= 0:
            return self.mid
        return (self.best_bid * qa + self.best_ask * qb) / total

    # -- shape ---------------------------------------------------------- #

    def depth(self, levels: int = 1) -> tuple[float, float]:
        """Summed (bid, ask) size over the top ``levels``."""
        return (
            sum(lvl.size for lvl in self.bids[:levels]),
            sum(lvl.size for lvl in self.asks[:levels]),
        )

    def imbalance(self, levels: int = 1) -> float:
        """(bid - ask) / (bid + ask) over the top ``levels``, in [-1, 1].

        Positive is bid-heavy. This is the single most predictive cheap
        microstructure feature at short horizons, and also the one most
        contaminated by spoofing beyond the touch — hence the explicit depth
        argument rather than a hidden default over the whole book.
        """
        qb, qa = self.depth(levels)
        total = qb + qa
        return 0.0 if total <= 0 else (qb - qa) / total

    def age_ms(self, now: pd.Timestamp) -> float:
        """Milliseconds since this snapshot was stamped. Negative if `now` precedes it."""
        return (now - self.ts).total_seconds() * 1e3
