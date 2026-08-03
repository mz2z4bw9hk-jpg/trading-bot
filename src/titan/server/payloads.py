"""Payload builders shared by the dashboard API and the static exporter.

The dashboard page consumes a fixed set of JSON payloads. The FastAPI app
serves each one per request; ``titan export`` bakes the same payloads into a
single self-contained HTML file. Keeping the builders here — and nowhere
else — guarantees the live view and the static export can never drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from titan.core.jsonsafe import json_safe

MAX_POINTS = 1500

# Full JSON API surface: route name -> artifact file.
JSON_ARTIFACTS = {
    "report": "report.json",
    "signals": "signals.json",
    "trades": "trades.json",
    "scan": "scan.json",
    "importance": "importance.json",
    "universe": "universe.json",
    "correlation": "correlation.json",
    "quality": "quality.json",
    "manifest": "manifest.json",
    "tracking": "tracking.json",
    "account": "account.json",
}

# What the dashboard page actually renders. ``signals``/``trades`` are audit
# artifacts consumed off-dashboard; embedding them would multiply the export
# size for pixels that never change.
DASHBOARD_KEYS = (
    "report",
    "scan",
    "importance",
    "universe",
    "correlation",
    "quality",
    "manifest",
    "tracking",
    "account",
    "equity",
    "regimes",
)


def downsample(df: pd.DataFrame, max_points: int = MAX_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(len(df) / max_points) + 1
    # Keep the last row: the most recent state matters most.
    return pd.concat([df.iloc[::step], df.tail(1)]).drop_duplicates()


def json_payload(artifacts: Path, name: str) -> object | None:
    """One of the plain JSON artifacts; ``None`` when it was never written.

    Sanitized on read as well as on write: artifacts produced before the
    write-side guard existed can contain ``NaN``/``Infinity`` tokens, which
    Python parses happily but ``JSONResponse`` will not serve.
    """
    path = artifacts / JSON_ARTIFACTS[name]
    if not path.exists():
        return None
    return json_safe(json.loads(path.read_text()))


def equity_payload(artifacts: Path) -> dict | None:
    path = artifacts / "equity.csv"
    if not path.exists():
        return None
    df = downsample(pd.read_csv(path, parse_dates=["date"]))
    return {
        "date": [str(d.date()) for d in df["date"]],
        "equity": [round(float(v), 2) for v in df["equity"]],
        "drawdown": [round(float(v), 5) for v in df["drawdown"]],
        "exposure": [round(float(v), 4) for v in df["exposure"]],
    }


def regimes_payload(artifacts: Path) -> dict | None:
    path = artifacts / "regimes.csv"
    if not path.exists():
        return None
    df = downsample(pd.read_csv(path, parse_dates=["date"]))
    return {
        "date": [str(d.date()) for d in df["date"]],
        "regime": df["regime"].astype(str).tolist(),
        "vol_state": df["vol_state"].astype(str).tolist(),
        "confidence": [round(float(v), 3) for v in df["confidence"]],
    }


def collect_payloads(
    artifacts: Path, keys: tuple[str, ...] = DASHBOARD_KEYS
) -> dict[str, object]:
    """Every payload the dashboard fetches, keyed exactly like the API routes.

    Artifacts that were never written map to ``None`` — the page renders its
    own empty states, so partial artifact dirs export without failing.
    """
    builders = {"equity": equity_payload, "regimes": regimes_payload}
    return {
        k: builders[k](artifacts) if k in builders else json_payload(artifacts, k)
        for k in keys
    }
