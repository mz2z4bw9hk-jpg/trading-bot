"""The pipeline's one safety property: nothing reaches execution around risk.

Python cannot seal a constructor, so the guarantee is enforced two ways that
together are checkable: ``ClearedOrder`` refuses to build without the mint
token, and this module fails the build if any file other than the risk agent
mentions that token. The second half is the one that matters — the first is
only as good as the fact that there is exactly one way to satisfy it.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest

from titan.agents.contracts import (
    ClearedOrder,
    Disposition,
    Intent,
    OrderSide,
    RiskBypassError,
    RiskVerdict,
)

SRC = pathlib.Path(__file__).resolve().parents[1] / "src" / "titan"
ALLOWED_MINT_REFERENCES = {"contracts.py", "risk_agent.py"}


def _intent(**kw) -> Intent:
    base = {
        "symbol": "ABC",
        "side": OrderSide.BUY,
        "edge_bps": 8.0,
        "horizon_s": 30.0,
        "confidence": 0.8,
        "ts": pd.Timestamp("2026-01-05 14:30:00", tz="UTC"),
        "source": "test",
    }
    return Intent(**{**base, **kw})


def test_the_mint_is_referenced_in_exactly_one_agent():
    """Grep the tree. A second construction site is a governance failure."""
    offenders = []
    for path in SRC.rglob("*.py"):
        if path.name in ALLOWED_MINT_REFERENCES:
            continue
        if "_MINT" in path.read_text():
            offenders.append(path.relative_to(SRC).as_posix())

    assert not offenders, (
        f"the risk mint is referenced outside the risk agent: {offenders}. "
        "Every order must be cleared by RiskAgent.clear()."
    )


def test_a_cleared_order_cannot_be_forged():
    with pytest.raises(RiskBypassError):
        ClearedOrder(
            mint=object(), intent=_intent(), qty=10.0,
            limit_price=None, verdict_id="forged",
        )


def test_an_intent_carries_no_size():
    """Sizing is a risk decision; the input to risk must not pre-empt it."""
    fields = set(Intent.__dataclass_fields__)
    for forbidden in ("qty", "size", "quantity", "notional", "limit_price"):
        assert forbidden not in fields, (
            f"Intent.{forbidden} would let the signal agent size its own orders"
        )


def test_a_rejected_verdict_cannot_be_cleared():
    from titan.agents.risk_agent import RiskAgent

    agent = RiskAgent()
    verdict = RiskVerdict(
        intent=_intent(),
        disposition=Disposition.REJECTED,
        approved_qty=0.0,
        binding_constraint="toxicity",
        detail="VPIN unmeasurable",
    )
    with pytest.raises(ValueError, match="nothing to clear"):
        agent.clear(verdict)


def test_a_verdict_from_another_agent_cannot_be_cleared():
    """Clearing requires the audit record to exist in THIS agent's log."""
    from titan.agents.risk_agent import RiskAgent

    agent = RiskAgent()
    smuggled = RiskVerdict(
        intent=_intent(),
        disposition=Disposition.APPROVED,
        approved_qty=100.0,
        binding_constraint="none",
        detail="hand-built",
    )
    with pytest.raises(ValueError, match="not in this agent's audit log"):
        agent.clear(smuggled)
