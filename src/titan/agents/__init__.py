"""Multi-agent trading pipeline: signal -> risk -> execution.

The boundary types live in :mod:`titan.agents.contracts`. Import those rather
than reaching across agents for concrete classes — the whole point of the
separation is that each agent can be replaced without its neighbours noticing.
"""

from titan.agents.contracts import (
    ClearedOrder,
    Disposition,
    Intent,
    OrderSide,
    RiskBypassError,
    RiskVerdict,
)
from titan.agents.governance import ProposalGate, ProtectedPathError
from titan.agents.risk_agent import PortfolioState, RiskAgent, RiskLimits

__all__ = [
    "ClearedOrder",
    "Disposition",
    "Intent",
    "OrderSide",
    "PortfolioState",
    "ProposalGate",
    "ProtectedPathError",
    "RiskAgent",
    "RiskBypassError",
    "RiskLimits",
    "RiskVerdict",
]
