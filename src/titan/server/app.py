"""Dashboard API: serves validated research artifacts, computes nothing.

Every endpoint is a read-through to files produced by ``titan validate`` /
``titan scan``. If the dashboard shows a number, that number exists in an
artifact on disk and can be audited. Missing artifacts return 404 with a hint
instead of empty fabrications. Payload construction lives in ``payloads.py``,
shared with the static exporter, so both views serve identical numbers.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, Response

from titan.server.payloads import (
    JSON_ARTIFACTS,
    equity_payload,
    json_payload,
    regimes_payload,
)

_STATIC = Path(__file__).parent / "static"


def create_app(artifacts_dir: str | Path) -> FastAPI:
    artifacts = Path(artifacts_dir)
    app = FastAPI(title="TITAN dashboard", docs_url=None, redoc_url=None)

    def _read_json(name: str) -> object:
        payload = json_payload(artifacts, name)
        if payload is None:
            raise HTTPException(
                status_code=404,
                detail=f"artifact {JSON_ARTIFACTS[name]} not found — run `titan validate` first",
            )
        return payload

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "artifacts": str(artifacts.resolve())}

    for name in JSON_ARTIFACTS:

        def _make(name: str = name):
            def endpoint() -> JSONResponse:
                return JSONResponse(_read_json(name))

            return endpoint

        app.get(f"/api/{name}", name=f"api_{name}")(_make())

    @app.get("/api/equity")
    def equity() -> JSONResponse:
        payload = equity_payload(artifacts)
        if payload is None:
            raise HTTPException(status_code=404, detail="equity.csv not found")
        return JSONResponse(payload)

    @app.get("/api/regimes")
    def regimes() -> JSONResponse:
        payload = regimes_payload(artifacts)
        if payload is None:
            raise HTTPException(status_code=404, detail="regimes.csv not found")
        return JSONResponse(payload)

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
