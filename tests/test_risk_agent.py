"""The risk agent's contract: fail closed, size it yourself, name the binder."""

from __future__ import annotations

import pandas as pd
import pytest

from titan.agents.contracts import Disposition, Intent, OrderSide
from titan.agents.risk_agent import PortfolioState, RiskAgent, RiskLimits
from titan.microstructure.book import BookLevel, BookSnapshot
from titan.microstructure.toxicity import ToxicityState

TS = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")


def _book(
    *, bid=99.95, ask=100.05, bid_size=5_000.0, ask_size=5_000.0, ts=TS, symbol="ABC"
) -> BookSnapshot:
    return BookSnapshot(
        symbol=symbol,
        ts=ts,
        bids=(BookLevel(bid, bid_size), BookLevel(bid - 0.05, bid_size)),
        asks=(BookLevel(ask, ask_size), BookLevel(ask + 0.05, ask_size)),
    )


def _intent(**kw) -> Intent:
    base = {
        "symbol": "ABC", "side": OrderSide.BUY, "edge_bps": 12.0,
        "horizon_s": 30.0, "confidence": 1.0, "ts": TS, "source": "stat-arb",
    }
    return Intent(**{**base, **kw})


def _tox(**kw) -> ToxicityState:
    base = {"symbol": "ABC", "vpin": 0.2, "ofi": 0.0, "kyle_lambda": 1e-9}
    return ToxicityState(**{**base, **kw})


def _clean_state() -> PortfolioState:
    return PortfolioState(positions={}, marks={"ABC": 100.0})


# --------------------------------------------------------------------------- #
# Fail closed
# --------------------------------------------------------------------------- #


def test_unmeasurable_toxicity_rejects():
    """The whole fail-closed rule in one case: nan is not 'fine'."""
    agent = RiskAgent()
    verdict = agent.evaluate(
        _intent(), _book(), _tox(vpin=float("nan")), _clean_state()
    )
    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "toxicity"
    assert "unmeasurable" in verdict.detail


def test_unmeasurable_price_impact_rejects():
    agent = RiskAgent()
    verdict = agent.evaluate(
        _intent(), _book(), _tox(kyle_lambda=float("nan")), _clean_state()
    )
    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "edge_vs_costs"


def test_a_check_that_raises_rejects_rather_than_propagates():
    """An exception inside risk must not become an unrisked order upstream."""
    agent = RiskAgent()

    def boom(*args, **kwargs):
        raise RuntimeError("estimator exploded")

    agent._check_toxicity = boom  # type: ignore[method-assign]
    verdict = agent.evaluate(_intent(), _book(), _tox(), _clean_state())

    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "internal_error"
    assert "estimator exploded" in verdict.detail


@pytest.mark.parametrize(
    "book,expected",
    [
        (_book(bid=100.10, ask=100.05), "crossed"),
        (_book(bid=100.00, ask=100.00), "locked"),
        (_book(ask=101.00), "spread"),
        (_book(ts=TS - pd.Timedelta(seconds=5)), "old"),
        (_book(ts=TS + pd.Timedelta(seconds=1)), "postdates"),
    ],
)
def test_book_integrity_triage(book, expected):
    agent = RiskAgent()
    verdict = agent.evaluate(_intent(), book, _tox(), _clean_state())
    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "book_integrity"
    assert expected in verdict.detail


def test_a_book_for_the_wrong_symbol_rejects():
    agent = RiskAgent()
    verdict = agent.evaluate(_intent(), _book(symbol="XYZ"), _tox(), _clean_state())
    assert verdict.binding_constraint == "book_integrity"


# --------------------------------------------------------------------------- #
# Hard limits
# --------------------------------------------------------------------------- #


def test_the_kill_switch_stops_new_risk_at_the_drawdown_limit():
    agent = RiskAgent(RiskLimits(equity=1_000_000, kill_drawdown_pct=4.0))
    state = PortfolioState(
        positions={}, marks={"ABC": 100.0},
        session_high_water=0.0, session_pnl=-45_000.0,
    )
    verdict = agent.evaluate(_intent(), _book(), _tox(), state)
    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "kill_switch"


def test_drawdown_tapers_size_before_it_halts():
    limits = RiskLimits(warn_drawdown_pct=2.0, kill_drawdown_pct=4.0)
    agent = RiskAgent(limits)
    flat = agent.evaluate(_intent(), _book(), _tox(), _clean_state())
    hurting = agent.evaluate(
        _intent(), _book(), _tox(),
        PortfolioState(marks={"ABC": 100.0}, session_pnl=-30_000.0),
    )
    assert hurting.is_cleared
    assert hurting.approved_qty < flat.approved_qty


def test_a_halted_symbol_rejects():
    agent = RiskAgent()
    state = PortfolioState(marks={"ABC": 100.0}, halted=frozenset({"ABC"}))
    verdict = agent.evaluate(_intent(), _book(), _tox(), state)
    assert verdict.binding_constraint == "kill_switch"


def test_position_limit_scales_rather_than_rejects_when_room_remains():
    limits = RiskLimits(max_position_notional=3_000.0, max_order_notional=1e9,
                        base_risk_fraction=0.01, min_order_notional=100.0)
    agent = RiskAgent(limits)
    state = PortfolioState(positions={"ABC": 20.0}, marks={"ABC": 100.0})

    verdict = agent.evaluate(_intent(), _book(), _tox(), state)

    assert verdict.disposition is Disposition.RESIZED
    assert verdict.binding_constraint == "position_limits"
    # 3_000 cap less the 2_000 already held leaves 1_000 of notional.
    assert verdict.approved_qty * 100.0 <= 1_000.0 + 1e-6


def test_a_full_position_can_still_be_reduced():
    """A symbol at its cap must remain closable, or limits create traps."""
    limits = RiskLimits(max_position_notional=2_000.0, base_risk_fraction=0.01,
                        min_order_notional=100.0)
    agent = RiskAgent(limits)
    state = PortfolioState(positions={"ABC": 20.0}, marks={"ABC": 100.0})

    closing = agent.evaluate(_intent(side=OrderSide.SELL), _book(), _tox(), state)

    assert closing.is_cleared, closing.detail


def test_participation_caps_the_order_against_the_far_touch():
    limits = RiskLimits(base_risk_fraction=0.5, max_order_notional=1e9,
                        max_position_notional=1e9, max_gross_notional=1e9,
                        max_net_notional=1e9, max_touch_participation=0.10)
    agent = RiskAgent(limits)
    book = _book(ask_size=100.0)

    verdict = agent.evaluate(_intent(), book, _tox(), _clean_state())

    assert verdict.is_cleared
    assert verdict.approved_qty <= 10.0 + 1e-9   # 10% of the 100 on the ask


# --------------------------------------------------------------------------- #
# Edge economics
# --------------------------------------------------------------------------- #


def test_an_edge_that_does_not_clear_the_spread_is_refused():
    agent = RiskAgent()
    thin = _intent(edge_bps=1.0)          # half-spread alone is ~5bps here
    verdict = agent.evaluate(thin, _book(), _tox(), _clean_state())

    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "edge_vs_costs"


def test_expected_impact_is_charged_against_the_edge():
    """Same edge, same book; only the impact coefficient differs."""
    agent = RiskAgent()
    cheap = agent.evaluate(_intent(), _book(), _tox(kyle_lambda=1e-9), _clean_state())
    costly = agent.evaluate(_intent(), _book(), _tox(kyle_lambda=1e-2), _clean_state())

    assert cheap.is_cleared
    assert costly.disposition is Disposition.REJECTED
    assert costly.binding_constraint == "edge_vs_costs"


def test_order_flow_pushing_against_the_intent_rejects():
    agent = RiskAgent(RiskLimits(max_adverse_ofi=2.0))
    # Sellers hitting the bid while we want to buy.
    verdict = agent.evaluate(_intent(), _book(), _tox(ofi=-5.0), _clean_state())
    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "toxicity"

    supportive = agent.evaluate(_intent(), _book(), _tox(ofi=5.0), _clean_state())
    assert supportive.is_cleared


def test_dust_orders_are_rejected_not_shipped():
    limits = RiskLimits(base_risk_fraction=1e-6, min_order_notional=500.0)
    agent = RiskAgent(limits)
    verdict = agent.evaluate(_intent(), _book(), _tox(), _clean_state())

    assert verdict.disposition is Disposition.REJECTED
    assert verdict.binding_constraint == "min_order_notional"


# --------------------------------------------------------------------------- #
# Sizing authority and audit
# --------------------------------------------------------------------------- #


def test_risk_sizes_the_order_from_confidence_alone():
    agent = RiskAgent(RiskLimits(max_order_notional=1e9, max_position_notional=1e9,
                                 max_touch_participation=1.0))
    strong = agent.evaluate(_intent(confidence=1.0), _book(), _tox(), _clean_state())
    weak = agent.evaluate(_intent(confidence=0.25), _book(), _tox(), _clean_state())

    assert strong.approved_qty > weak.approved_qty > 0


def test_every_check_is_retained_including_the_ones_that_passed():
    agent = RiskAgent()
    verdict = agent.evaluate(_intent(), _book(), _tox(), _clean_state())

    names = [c.name for c in verdict.checks]
    assert names == [
        "kill_switch", "book_integrity", "toxicity",
        "edge_vs_costs", "position_limits", "participation",
    ]
    assert all(c.passed for c in verdict.checks)


def test_the_binding_constraint_is_the_smallest_scale():
    limits = RiskLimits(base_risk_fraction=0.5, max_order_notional=1e9,
                        max_position_notional=1e9, max_gross_notional=1e9,
                        max_net_notional=1e9, max_touch_participation=0.01)
    agent = RiskAgent(limits)
    verdict = agent.evaluate(_intent(), _book(ask_size=50_000.0), _tox(), _clean_state())

    assert verdict.binding_constraint == "participation"
    scales = {c.name: c.scale for c in verdict.checks}
    assert scales["participation"] == min(scales.values())


def test_every_decision_lands_in_the_audit_log():
    agent = RiskAgent()
    agent.evaluate(_intent(), _book(), _tox(), _clean_state())
    agent.evaluate(_intent(edge_bps=0.1), _book(), _tox(), _clean_state())

    assert len(agent.audit_log) == 2
    assert {v.disposition for v in agent.audit_log} == {
        Disposition.APPROVED, Disposition.REJECTED
    }
    assert all(v.to_dict()["checks"] for v in agent.audit_log)


def test_clearing_produces_an_order_tied_to_its_verdict():
    agent = RiskAgent()
    verdict = agent.evaluate(_intent(), _book(), _tox(), _clean_state())
    order = agent.clear(verdict, limit_price=99.95)

    assert order.verdict_id == verdict.verdict_id
    assert order.qty == verdict.approved_qty
    assert order.symbol == "ABC"
    assert order.side is OrderSide.BUY
