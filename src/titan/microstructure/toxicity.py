"""Toxicity estimators: is the flow we are about to trade against informed?

Three standard measures, each answering a different question:

- **Order flow imbalance** (Cont-Kukanov-Stoikov): net pressure at the touch
  between two book snapshots. Fast, and the one that moves within a quote.
- **VPIN** (Easley-Lopez de Prado-O'Hara): the fraction of volume that is
  one-sided, measured in volume time rather than clock time. Slow, and the one
  that flags a regime where making markets is a losing proposition.
- **Kyle's lambda**: the price impact coefficient — how far the mid moves per
  unit of signed volume. Converts a size into an expected adverse move, which
  is what the risk agent actually needs to compare against an edge.

Every estimator returns ``nan`` rather than a number when its history is too
short. A toxicity estimate that silently degrades to zero on thin data is
worse than none: zero reads as "safe" at exactly the moment the measurement
has nothing behind it.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
from scipy.stats import norm

from titan.microstructure.book import BookSnapshot

__all__ = ["ToxicityState", "kyle_lambda", "order_flow_imbalance", "rolling_ofi", "vpin"]


def order_flow_imbalance(prev: BookSnapshot, curr: BookSnapshot) -> float:
    """Signed size added at the touch between two snapshots.

    The Cont-Kukanov-Stoikov contribution. Each side contributes the size
    *added* when its price improves or holds, and the size *removed* when it
    retreats:

        e = 1{Pb_t >= Pb_t-1} Qb_t - 1{Pb_t <= Pb_t-1} Qb_t-1
          - 1{Pa_t <= Pa_t-1} Qa_t + 1{Pa_t >= Pa_t-1} Qa_t-1

    Positive means net buying pressure. Note both indicators fire when a price
    is unchanged, which is intended: the term then reduces to the size delta.
    """
    if prev.is_empty or curr.is_empty:
        return float("nan")

    pb0, pb1 = prev.best_bid, curr.best_bid
    qb0, qb1 = prev.bids[0].size, curr.bids[0].size
    pa0, pa1 = prev.best_ask, curr.best_ask
    qa0, qa1 = prev.asks[0].size, curr.asks[0].size

    bid_term = (qb1 if pb1 >= pb0 else 0.0) - (qb0 if pb1 <= pb0 else 0.0)
    ask_term = (qa1 if pa1 <= pa0 else 0.0) - (qa0 if pa1 >= pa0 else 0.0)
    return float(bid_term - ask_term)


def rolling_ofi(snapshots: list[BookSnapshot], window: int = 50) -> float:
    """Summed OFI over the last ``window`` snapshot transitions.

    Normalised by the mean touch size over the same window, so the number is
    comparable across instruments that quote in different lot conventions —
    a raw share count says nothing without knowing what a normal quote is.
    """
    if len(snapshots) < 3:
        return float("nan")
    tail = snapshots[-(window + 1):]
    events = [
        order_flow_imbalance(a, b) for a, b in pairwise(tail)
    ]
    finite = [e for e in events if np.isfinite(e)]
    if not finite:
        return float("nan")

    sizes = [
        0.5 * (s.bids[0].size + s.asks[0].size) for s in tail if not s.is_empty
    ]
    scale = float(np.mean(sizes)) if sizes else 0.0
    if scale <= 0:
        return float("nan")
    return float(np.sum(finite) / (scale * len(finite)))


def vpin(
    prices: np.ndarray,
    volumes: np.ndarray,
    *,
    bucket_volume: float,
    n_buckets: int = 50,
) -> float:
    """Volume-synchronised probability of informed trading, in [0, 1].

    Trades are bucketed by cumulative volume rather than by clock, then each
    bucket's volume is split into buy and sell parts by bulk volume
    classification — the normal CDF of the standardised price change over the
    bucket. VPIN is the mean absolute imbalance across the last ``n_buckets``.

    Values near 0 mean two-sided flow (a market maker's friend); values near 1
    mean the buckets are one-directional, which is what order flow looks like
    when someone knows something.
    """
    if bucket_volume <= 0:
        raise ValueError("bucket_volume must be positive")
    prices = np.asarray(prices, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    if prices.shape != volumes.shape:
        raise ValueError("prices and volumes must align")
    if len(prices) < 3 or volumes.sum() < bucket_volume * 2:
        return float("nan")

    # Bucket boundaries in volume time.
    cum = np.cumsum(volumes)
    edges = np.arange(bucket_volume, cum[-1], bucket_volume)
    idx = np.searchsorted(cum, edges)
    if len(idx) < 2:
        return float("nan")

    bucket_prices = prices[np.clip(idx, 0, len(prices) - 1)]
    dp = np.diff(bucket_prices)
    sigma = float(np.std(dp))
    if sigma <= 0:
        # No price variation across buckets: flow is perfectly balanced by
        # this measure's own logic, not unmeasurable.
        return 0.0

    buy_frac = norm.cdf(dp / sigma)
    imbalance = np.abs(2.0 * buy_frac - 1.0)
    tail = imbalance[-n_buckets:]
    return float(np.mean(tail))


def kyle_lambda(
    mid_changes: np.ndarray, signed_volume: np.ndarray, *, min_obs: int = 30
) -> float:
    """Price impact per unit of signed volume, from a no-intercept regression.

    ``lambda`` in dP = lambda * Q. Units are price per unit volume, so multiply
    by an order's size to get the adverse move that order should expect to
    cause. No intercept: a zero net order should imply no price change, and
    fitting one lets the regression absorb drift into a constant that has no
    microstructural meaning.
    """
    dp = np.asarray(mid_changes, dtype=float)
    q = np.asarray(signed_volume, dtype=float)
    if dp.shape != q.shape:
        raise ValueError("mid_changes and signed_volume must align")

    ok = np.isfinite(dp) & np.isfinite(q)
    dp, q = dp[ok], q[ok]
    if len(dp) < min_obs:
        return float("nan")
    denom = float(q @ q)
    if denom <= 0:
        return float("nan")
    return float((q @ dp) / denom)


@dataclass(frozen=True, slots=True)
class ToxicityState:
    """What the risk agent is handed. Any field may be ``nan``.

    ``nan`` is not "fine" — :class:`~titan.agents.risk_agent.RiskAgent` treats
    an unmeasurable toxicity as a rejection, because the alternative is quoting
    into flow you have decided not to look at.
    """

    symbol: str
    vpin: float = float("nan")
    ofi: float = float("nan")
    kyle_lambda: float = float("nan")

    def expected_adverse_bps(self, qty: float, mid: float) -> float:
        """Adverse selection this order should expect to pay, in bps.

        Kyle's lambda gives the mid move the order itself causes. A maker is
        filled *against* that move, so it is a cost regardless of side, and it
        is returned unsigned.
        """
        if not np.isfinite(self.kyle_lambda) or mid <= 0:
            return float("nan")
        return abs(1e4 * self.kyle_lambda * qty / mid)
