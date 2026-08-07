"""Spread capture versus toxicity: the decomposition, and what it must catch.

The identity under test throughout:

    gross spread capture = realized spread + adverse selection

If that ever fails to hold to floating-point tolerance, every number the
evaluator reports is describing something other than the trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.agents.contracts import OrderSide
from titan.execution.evaluation import (
    ExecutionEvaluator,
    Fill,
    MidTimeline,
)

T0 = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")


def _timeline(points: dict[float, float], symbol: str = "ABC") -> dict[str, MidTimeline]:
    """Mids at offsets in seconds from T0."""
    idx = pd.DatetimeIndex([T0 + pd.Timedelta(seconds=s) for s in sorted(points)])
    return {symbol: MidTimeline(pd.Series([points[s] for s in sorted(points)], index=idx))}


def _flat_timeline(seconds: int = 700, mid: float = 100.0) -> dict[str, MidTimeline]:
    return _timeline({float(s): mid for s in range(0, seconds)})


def _fill(**kw) -> Fill:
    base = {
        "symbol": "ABC", "ts": T0 + pd.Timedelta(seconds=10),
        "side": OrderSide.BUY, "price": 99.95, "qty": 100.0,
        "liquidity": "maker", "fee_bps": 0.0, "venue": "XNAS",
    }
    return Fill(**{**base, **kw})


# --------------------------------------------------------------------------- #
# The identity
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("side", [OrderSide.BUY, OrderSide.SELL])
@pytest.mark.parametrize("drift", [-0.30, -0.05, 0.0, 0.05, 0.30])
def test_capture_equals_realized_plus_adverse_selection(side, drift):
    """Holds for both sides and any subsequent move. This is the whole model."""
    mids = {float(s): 100.0 for s in range(0, 12)}
    mids.update({float(s): 100.0 + drift for s in range(12, 700)})
    price = 99.95 if side is OrderSide.BUY else 100.05

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(1.0, 30.0, 300.0))
    ev = evaluator.evaluate_fill(_fill(side=side, price=price))

    for horizon, m in ev.markouts.items():
        assert m.measured, horizon
        assert np.isclose(
            m.gross_capture_bps, m.realized_spread_bps + m.adverse_selection_bps
        ), f"identity broken at {horizon}s"


def test_a_maker_buying_below_the_mid_captures_spread():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    ev = evaluator.evaluate_fill(_fill(side=OrderSide.BUY, price=99.95))

    m = ev.markouts[30.0]
    assert m.gross_capture_bps == pytest.approx(5.0, abs=1e-9)   # 5bps of a 100 mid
    assert m.adverse_selection_bps == pytest.approx(0.0, abs=1e-9)
    assert m.realized_spread_bps == pytest.approx(5.0, abs=1e-9)


def test_a_maker_selling_above_the_mid_captures_spread():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    ev = evaluator.evaluate_fill(_fill(side=OrderSide.SELL, price=100.05))

    assert ev.markouts[30.0].gross_capture_bps == pytest.approx(5.0, abs=1e-9)


def test_crossing_the_spread_shows_as_negative_capture():
    """A taker pays the spread; the sign must say so without special-casing."""
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    ev = evaluator.evaluate_fill(
        _fill(side=OrderSide.BUY, price=100.05, liquidity="taker")
    )
    assert ev.markouts[30.0].gross_capture_bps == pytest.approx(-5.0, abs=1e-9)


# --------------------------------------------------------------------------- #
# Toxicity
# --------------------------------------------------------------------------- #


def test_getting_picked_off_shows_as_adverse_selection():
    """Bought at the bid, then the market left without us."""
    mids = {float(s): 100.0 for s in range(0, 11)}
    mids.update({float(s): 99.80 for s in range(11, 700)})

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(30.0,))
    m = evaluator.evaluate_fill(_fill(price=99.95)).markouts[30.0]

    assert m.gross_capture_bps == pytest.approx(5.0, abs=1e-9)
    assert m.adverse_selection_bps == pytest.approx(20.0, abs=1e-9)
    assert m.realized_spread_bps == pytest.approx(-15.0, abs=1e-9)
    assert m.toxicity_ratio == pytest.approx(4.0, abs=1e-9)


def test_the_verdict_reads_the_net_not_the_capture():
    """The failure mode this whole module exists to prevent.

    Every fill captures a positive spread. The desk is still losing money,
    because the mid runs away after each one. A capture-only dashboard shows
    green here.
    """
    mids = {float(s): 100.0 for s in range(0, 11)}
    mids.update({float(s): 99.85 for s in range(11, 700)})

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(30.0,), decision_horizon_s=30.0)
    report = evaluator.run([_fill(price=99.95)])

    assert report.decision.gross_capture_bps > 0
    assert report.decision.net_bps < 0
    assert report.verdict().startswith("UNPROFITABLE")


def test_a_healthy_book_is_reported_healthy():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    report = evaluator.run([_fill(price=99.95) for _ in range(5)])

    assert report.verdict().startswith("HEALTHY")
    assert report.decision.net_bps == pytest.approx(5.0, abs=1e-9)


def test_a_thin_margin_is_flagged_before_it_turns_negative():
    mids = {float(s): 100.0 for s in range(0, 11)}
    mids.update({float(s): 99.96 for s in range(11, 700)})

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(30.0,))
    report = evaluator.run([_fill(price=99.95)])

    assert report.decision.net_bps > 0
    assert report.verdict().startswith("THIN")


# --------------------------------------------------------------------------- #
# Aggregation
# --------------------------------------------------------------------------- #


def test_aggregates_are_notional_weighted():
    """Small winners must not outvote a large loser.

    Nine 1-lot fills capture spread cleanly; one 1000-lot fill is run over.
    Equal weighting reports a profitable desk. Notional weighting does not.
    """
    mids = {float(s): 100.0 for s in range(0, 700)}
    good = [
        Fill(symbol="ABC", ts=T0 + pd.Timedelta(seconds=1), side=OrderSide.BUY,
             price=99.95, qty=1.0)
        for _ in range(9)
    ]
    toxic_mids = dict(mids)
    toxic_mids.update({float(s): 99.00 for s in range(300, 700)})
    big = Fill(symbol="ABC", ts=T0 + pd.Timedelta(seconds=290), side=OrderSide.BUY,
               price=99.95, qty=1000.0)

    evaluator = ExecutionEvaluator(_timeline(toxic_mids), horizons=(30.0,))
    report = evaluator.run([*good, big])

    assert report.decision.net_bps < 0, "notional weighting did not dominate"
    assert report.decision.net_bps_median > 0, "the median should still look fine"


def test_horizons_separate_latency_toxicity_from_information_toxicity():
    """Picked off instantly, then it comes back: a queue problem, not alpha."""
    mids = {float(s): 100.0 for s in range(0, 11)}
    mids.update({float(s): 99.85 for s in range(11, 40)})
    mids.update({float(s): 100.0 for s in range(40, 700)})

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(1.0, 5.0, 300.0))
    report = evaluator.run([_fill(price=99.95)])

    assert report.stats[5.0].adverse_selection_bps > 10.0
    assert report.stats[300.0].adverse_selection_bps == pytest.approx(0.0, abs=1e-9)


def test_bucketing_isolates_the_venue_that_is_hurting_us():
    """One venue fills into calm, the other only ever fills before a move."""
    mids = {float(s): 100.0 for s in range(0, 200)}
    mids.update({float(s): 99.80 for s in range(200, 700)})

    evaluator = ExecutionEvaluator(_timeline(mids), horizons=(30.0,))
    report = evaluator.run([
        _fill(venue="GOOD", price=99.95, ts=T0 + pd.Timedelta(seconds=10)),
        _fill(venue="BAD", price=99.95, ts=T0 + pd.Timedelta(seconds=190)),
    ])

    venues = report.by_bucket["venue"]
    assert set(venues) == {"GOOD", "BAD"}
    assert venues["GOOD"].net_bps > 0
    assert venues["BAD"].net_bps < 0
    # Aggregated, the two nearly cancel — which is why the bucket exists.
    assert abs(report.decision.net_bps) < abs(venues["BAD"].net_bps)


def test_fees_come_out_of_the_net_and_a_rebate_adds_to_it():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    charged = evaluator.evaluate_fill(_fill(fee_bps=2.0)).markouts[30.0]
    rebated = evaluator.evaluate_fill(_fill(fee_bps=-0.5)).markouts[30.0]

    assert charged.net_bps == pytest.approx(3.0, abs=1e-9)
    assert rebated.net_bps == pytest.approx(5.5, abs=1e-9)
    # The rebate must not contaminate the microstructure quantities.
    assert charged.realized_spread_bps == rebated.realized_spread_bps


# --------------------------------------------------------------------------- #
# Unmeasured is not zero
# --------------------------------------------------------------------------- #


def test_a_horizon_past_the_end_of_the_data_is_unmeasured_not_flat():
    """Averaging an unmeasurable markout as zero biases toxicity toward zero."""
    evaluator = ExecutionEvaluator(_flat_timeline(seconds=20), horizons=(1.0, 300.0))
    ev = evaluator.evaluate_fill(_fill())

    assert ev.markouts[1.0].measured
    assert not ev.markouts[300.0].measured
    assert np.isnan(ev.markouts[300.0].adverse_selection_bps)


def test_unmeasured_fills_are_excluded_from_aggregates_and_counted():
    evaluator = ExecutionEvaluator(
        _flat_timeline(seconds=20), horizons=(1.0, 300.0), decision_horizon_s=300.0
    )
    report = evaluator.run([_fill(), _fill()])

    assert report.n_fills == 2
    assert report.n_unmeasured == 2
    assert report.decision.n == 0
    assert report.verdict().startswith("NO DATA")


def test_a_fill_before_the_first_quote_is_not_evaluated():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    ev = evaluator.evaluate_fill(_fill(ts=T0 - pd.Timedelta(seconds=60)))
    assert ev.markouts == {}


def test_the_mid_lookup_never_reads_the_future():
    """asof semantics: a markout that peeks forward measures its own answer."""
    mids = _timeline({0.0: 100.0, 10.0: 105.0})["ABC"]
    assert mids.at(T0 + pd.Timedelta(seconds=9)) == 100.0
    assert mids.at(T0 + pd.Timedelta(seconds=10)) == 105.0


def test_a_symbol_with_no_timeline_is_unmeasured_rather_than_assumed():
    evaluator = ExecutionEvaluator(_flat_timeline(), horizons=(30.0,))
    ev = evaluator.evaluate_fill(_fill(symbol="NOPE"))
    assert ev.markouts == {}
