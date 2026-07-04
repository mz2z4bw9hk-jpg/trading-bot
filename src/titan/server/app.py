"""Dashboard API: serves validated research artifacts, computes nothing.

Every endpoint is a read-through to files produced by ``titan validate`` /
``titan scan``. If the dashboard shows a number, that number exists in an
artifact on disk and can be audited. Missing artifacts return 404 with a hint
instead of empty fabrications.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

_STATIC = Path(__file__).parent / "static"
_MAX_POINTS = 1500

_JSON_ARTIFACTS = {
    "report": "report.json",
    "signals": "signals.json",
    "trades": "trades.json",
    "scan": "scan.json",
    "importance": "importance.json",
    "universe": "universe.json",
    "correlation": "correlation.json",
    "quality": "quality.json",
    "manifest": "manifest.json",
}


def _downsample(df: pd.DataFrame, max_points: int = _MAX_POINTS) -> pd.DataFrame:
    if len(df) <= max_points:
        return df
    step = int(len(df) / max_points) + 1
    # Keep the last row: the most recent state matters most.
    return pd.concat([df.iloc[::step], df.tail(1)]).drop_duplicates()


def create_app(artifacts_dir: str | Path) -> FastAPI:
    artifacts = Path(artifacts_dir)
    app = FastAPI(title="TITAN dashboard", docs_url=None, redoc_url=None)

    def _read_json(name: str) -> object:
        path = artifacts / _JSON_ARTIFACTS[name]
        if not path.exists():
            raise HTTPException(
                status_code=404,
                detail=f"artifact {path.name} not found — run `titan validate` first",
            )
        return json.loads(path.read_text())

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "artifacts": str(artifacts.resolve())}

    for name in _JSON_ARTIFACTS:

        def _make(name: str = name):
            def endpoint() -> JSONResponse:
                return JSONResponse(_read_json(name))

            return endpoint

        app.get(f"/api/{name}", name=f"api_{name}")(_make())

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        path = artifacts / "equity.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="equity.csv not found")
        df = pd.read_csv(path, parse_dates=["date"])
        df = _downsample(df)
        return JSONResponse(
            {
                "date": [str(d.date()) for d in df["date"]],
                "equity": [round(float(v), 2) for v in df["equity"]],
                "drawdown": [round(float(v), 5) for v in df["drawdown"]],
                "exposure": [round(float(v), 4) for v in df["exposure"]],
            }
        )

    @app.get("/api/regimes")
    def regimes() -> JSONResponse:
        path = artifacts / "regimes.csv"
        if not path.exists():
            raise HTTPException(status_code=404, detail="regimes.csv not found")
        df = pd.read_csv(path, parse_dates=["date"])
        df = _downsample(df)
        return JSONResponse(
            {
                "date": [str(d.date()) for d in df["date"]],
                "regime": df["regime"].astype(str).tolist(),
                "vol_state": df["vol_state"].astype(str).tolist(),
                "confidence": [round(float(v), 3) for v in df["confidence"]],
            }
        )

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        page = _STATIC / "dashboard.html"
        if not page.exists():
            raise HTTPException(status_code=500, detail="dashboard.html missing from package")
        return page.read_text()

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
            '<rect width="16" height="16" rx="3" fill="#2a78d6"/>'
            '<path d="M3 5h10M8 5v7" stroke="#fff" stroke-width="2"/></svg>'
        )
        return Response(content=svg, media_type="image/svg+xml")

    return app
