"""Non-finite floats must never reach an artifact or the dashboard API.

A run that takes zero trades leaves a flat equity curve, whose skew and
kurtosis are NaN and whose tail ratio is +inf. Python's json writes those as
bare ``NaN``/``Infinity`` tokens, which are not valid JSON; FastAPI then
refuses to serve the artifact and the dashboard panel hangs on "loading…".
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from titan.artifacts import _write_json
from titan.backtest.metrics import summarize
from titan.core.jsonsafe import json_safe
from titan.server.app import create_app


def _strict(text: str) -> object:
    """Parse rejecting NaN/Infinity, the way a JSON-spec consumer would."""
    def _reject(token: str) -> object:
        raise ValueError(f"non-standard JSON constant: {token}")

    return json.loads(text, parse_constant=_reject)


def test_json_safe_nulls_non_finite_floats_recursively():
    payload = {
        "nan": float("nan"),
        "pos_inf": float("inf"),
        "neg_inf": float("-inf"),
        "finite": 1.5,
        "nested": [{"deep": float("nan")}, 2.0],
        "np32": np.float32("nan"),
        "np64": np.float64("inf"),
    }
    assert json_safe(payload) == {
        "nan": None,
        "pos_inf": None,
        "neg_inf": None,
        "finite": 1.5,
        "nested": [{"deep": None}, 2.0],
        "np32": None,
        "np64": None,
    }


def test_json_safe_preserves_non_float_payloads():
    payload = {"s": "text", "i": 3, "b": True, "n": None, "l": [1, "two"]}
    assert json_safe(payload) == payload


def test_zero_trade_summary_produces_non_finite_metrics():
    """The condition this guard exists for is real, not hypothetical."""
    summary = dataclasses.asdict(summarize(pd.Series([1e6] * 600), trades=[]))
    non_finite = {
        k for k, v in summary.items() if isinstance(v, float) and not np.isfinite(v)
    }
    assert non_finite, "expected a flat equity curve to yield NaN/inf metrics"


def test_written_artifact_is_strict_json(tmp_path):
    summary = dataclasses.asdict(summarize(pd.Series([1e6] * 600), trades=[]))
    path = tmp_path / "report.json"
    _write_json(path, {"backtest": {"summary": summary}})

    parsed = _strict(path.read_text())  # would raise on NaN/Infinity
    assert parsed["backtest"]["summary"]["tail_ratio"] is None


def test_api_serves_a_report_containing_nan_tokens(tmp_path):
    """Artifacts written before this guard must still serve."""
    (tmp_path / "report.json").write_text(
        json.dumps({"backtest": {"summary": {"sharpe": float("nan")}}})
    )
    with pytest.raises(ValueError):  # confirm the file really is non-standard
        _strict((tmp_path / "report.json").read_text())

    client = TestClient(create_app(tmp_path))
    response = client.get("/api/report")

    assert response.status_code == 200
    assert response.json()["backtest"]["summary"]["sharpe"] is None
