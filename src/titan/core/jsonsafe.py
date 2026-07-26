"""Strict-JSON sanitation for everything the platform serializes.

``NaN`` and ``±Infinity`` are legal in Python's ``json`` module but are *not*
valid JSON, and FastAPI's ``JSONResponse`` (``allow_nan=False``) refuses them
outright — a 500 that the dashboard cannot distinguish from a slow fetch.

They arise from perfectly correct arithmetic on degenerate runs: the skew and
kurtosis of a flat equity curve (a run that took zero trades), a profit factor
with no losing trade to divide by. The honest encoding for "undefined" is
``null``, which every formatter on the dashboard already renders as an em dash.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def json_safe(payload: Any) -> Any:
    """Recursively replace non-finite floats with ``None``.

    Applied on the way out (artifact writing) and on the way in (serving),
    so an artifact written before this guard existed still serves cleanly.
    """
    if isinstance(payload, float):  # np.float64 subclasses float
        return payload if math.isfinite(payload) else None
    if isinstance(payload, np.floating):  # float32/float16 do not
        value = float(payload)
        return value if math.isfinite(value) else None
    if isinstance(payload, dict):
        return {k: json_safe(v) for k, v in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [json_safe(v) for v in payload]
    return payload
