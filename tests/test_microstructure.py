"""Book state and toxicity estimators.

The recurring theme: an estimator with too little history returns ``nan``, not
a number. Downstream, ``nan`` rejects. Any estimator that quietly returns 0.0
on thin data reads as "safe" exactly when it knows nothing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.microstructure.book import BookLevel, BookSnapshot
from titan.microstructure.toxicity import (
    ToxicityState,
    kyle_lambda,
    order_flow_imbalance,
    rolling_ofi,
    vpin,
)

TS = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")


def _book(bid=99.95, ask=100.05, bq=1000.0, aq=1000.0, ts=TS) -> BookSnapshot:
    return BookSnapshot(
        symbol="ABC", ts=ts,
        bids=(BookLevel(bid, bq), BookLevel(bid - 0.01, bq)),
        asks=(BookLevel(ask, aq), BookLevel(ask + 0.01, aq)),
    )


# --------------------------------------------------------------------------- #
# Book
# --------------------------------------------------------------------------- #


def test_mid_and_spread():
    b = _book()
    assert b.mid == pytest.approx(100.0)
    assert b.spread == pytest.approx(0.10)
    assert b.spread_bps == pytest.approx(10.0)


def test_microprice_leans_toward_the_thin_side():
    """Heavy bid means buyers queueing: fair value sits above the mid."""
    heavy_bid = _book(bq=9000.0, aq=1000.0)
    assert heavy_bid.microprice > heavy_bid.mid

    heavy_ask = _book(bq=1000.0, aq=9000.0)
    assert heavy_ask.microprice < heavy_ask.mid

    assert _book().microprice == pytest.approx(_book().mid)


def test_imbalance_is_signed_and_bounded():
    assert _book(bq=1000.0, aq=1000.0).imbalance() == pytest.approx(0.0)
    assert _book(bq=1000.0, aq=0.0).imbalance() == pytest.approx(1.0)
    assert _book(bq=0.0, aq=1000.0).imbalance() == pytest.approx(-1.0)


def test_crossed_and_locked_books_are_detected_not_repaired():
    assert _book(bid=100.10, ask=100.05).is_crossed
    assert _book(bid=100.00, ask=100.00).is_locked
    assert not _book().is_crossed


def test_unordered_levels_are_reported():
    bad = BookSnapshot(
        symbol="ABC", ts=TS,
        bids=(BookLevel(99.95, 100.0), BookLevel(99.99, 100.0)),   # ascending: wrong
        asks=(BookLevel(100.05, 100.0),),
    )
    assert not bad.is_ordered


def test_a_level_with_a_nonsense_price_is_refused_at_construction():
    with pytest.raises(ValueError):
        BookLevel(0.0, 100.0)
    with pytest.raises(ValueError):
        BookLevel(float("nan"), 100.0)
    with pytest.raises(ValueError):
        BookLevel(100.0, -1.0)


# --------------------------------------------------------------------------- #
# Order flow imbalance
# --------------------------------------------------------------------------- #


def test_size_added_to_the_bid_is_positive_flow():
    prev, curr = _book(bq=1000.0), _book(bq=1500.0)
    assert order_flow_imbalance(prev, curr) == pytest.approx(500.0)


def test_size_added_to_the_ask_is_negative_flow():
    prev, curr = _book(aq=1000.0), _book(aq=1500.0)
    assert order_flow_imbalance(prev, curr) == pytest.approx(-500.0)


def test_a_bid_that_steps_up_is_strong_buying():
    prev = _book(bid=99.95, bq=1000.0)
    curr = _book(bid=99.99, bq=800.0)
    # New, higher bid counts in full; the retreating old level is not subtracted.
    assert order_flow_imbalance(prev, curr) == pytest.approx(800.0)


def test_ofi_is_symmetric_under_swapping_the_sides():
    up = order_flow_imbalance(_book(bq=1000.0), _book(bq=1500.0))
    down = order_flow_imbalance(_book(aq=1000.0), _book(aq=1500.0))
    assert up == pytest.approx(-down)


def test_rolling_ofi_needs_history():
    assert np.isnan(rolling_ofi([_book()]))
    assert np.isnan(rolling_ofi([]))


def test_rolling_ofi_normalises_by_typical_touch_size():
    """Same flow in different lot conventions must read the same."""
    small = [_book(bq=100.0 + 10 * i, aq=100.0) for i in range(20)]
    large = [_book(bq=10_000.0 + 1000 * i, aq=10_000.0) for i in range(20)]
    assert rolling_ofi(small) == pytest.approx(rolling_ofi(large), rel=1e-9)


# --------------------------------------------------------------------------- #
# VPIN
# --------------------------------------------------------------------------- #


def test_vpin_is_high_for_one_directional_flow():
    n = 400
    trending = np.cumsum(np.full(n, 0.05)) + 100.0
    volumes = np.full(n, 100.0)
    assert vpin(trending, volumes, bucket_volume=1000.0) > 0.9


def test_vpin_is_low_for_two_sided_flow():
    rng = np.random.default_rng(3)
    n = 400
    choppy = 100.0 + np.cumsum(rng.normal(0, 0.05, n))
    volumes = np.full(n, 100.0)
    assert vpin(choppy, volumes, bucket_volume=1000.0) < 0.75


def test_vpin_returns_nan_on_insufficient_volume():
    assert np.isnan(vpin(np.array([100.0, 101.0]), np.array([1.0, 1.0]), bucket_volume=1e6))


def test_vpin_rejects_a_nonpositive_bucket():
    with pytest.raises(ValueError):
        vpin(np.arange(10.0), np.ones(10), bucket_volume=0.0)


# --------------------------------------------------------------------------- #
# Kyle's lambda
# --------------------------------------------------------------------------- #


def test_kyle_lambda_recovers_a_known_impact_coefficient():
    rng = np.random.default_rng(0)
    q = rng.normal(0, 1000.0, 500)
    true_lambda = 2.5e-5
    dp = true_lambda * q + rng.normal(0, 1e-4, 500)

    assert kyle_lambda(dp, q) == pytest.approx(true_lambda, rel=0.05)


def test_kyle_lambda_needs_observations():
    assert np.isnan(kyle_lambda(np.zeros(5), np.zeros(5)))


def test_expected_adverse_scales_with_size_and_is_unsigned():
    tox = ToxicityState(symbol="ABC", kyle_lambda=1e-4)
    small = tox.expected_adverse_bps(100.0, 100.0)
    large = tox.expected_adverse_bps(1000.0, 100.0)

    assert large == pytest.approx(10 * small)
    assert small > 0
    # Direction of the coefficient does not change that impact is a cost.
    flipped = ToxicityState(symbol="ABC", kyle_lambda=-1e-4)
    assert flipped.expected_adverse_bps(100.0, 100.0) == pytest.approx(small)


def test_expected_adverse_is_nan_without_a_lambda():
    assert np.isnan(ToxicityState(symbol="ABC").expected_adverse_bps(100.0, 100.0))
