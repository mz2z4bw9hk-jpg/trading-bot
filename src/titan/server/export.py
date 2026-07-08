"""Static dashboard export: one self-contained HTML file, no server.

``titan export`` takes the artifacts a research run wrote and bakes every
payload the dashboard would fetch into the page itself. The result opens
with a double-click, attaches to an e-mail, and drops onto any static host —
no Python, no dependencies, no network calls. It is a frozen snapshot of one
run, not a live view: re-export after the next ``titan validate``.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from titan.server.payloads import collect_payloads

_TEMPLATE = Path(__file__).parent / "static" / "dashboard.html"
_MARKER = "<!-- TITAN:EMBED -->"


def render_static_dashboard(artifacts_dir: str | Path) -> tuple[str, dict[str, object]]:
    """Render the dashboard template with artifact payloads baked in.

    Returns the HTML text and the payload map (``None`` values = artifacts
    that were missing and therefore not embedded).
    """
    artifacts = Path(artifacts_dir)
    payloads = collect_payloads(artifacts)
    if all(v is None for v in payloads.values()):
        raise FileNotFoundError(
            f"no artifacts in {artifacts.resolve()} — run `titan validate` first"
        )
    template = _TEMPLATE.read_text()
    if _MARKER not in template:
        raise RuntimeError(f"embed marker {_MARKER!r} missing from dashboard template")

    embedded: dict[str, object] = {k: v for k, v in payloads.items() if v is not None}
    embedded["__export"] = {
        "exported_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "artifacts_dir": str(artifacts.resolve()),
    }
    # A literal "</" inside the JSON would terminate the <script> element
    # mid-blob; the escape "<\/" is byte-identical after JSON parsing.
    blob = json.dumps(embedded, separators=(",", ":")).replace("</", "<\\/")
    html = template.replace(_MARKER, f"<script>window.TITAN_EMBEDDED = {blob};</script>")
    return html, payloads


def export_static_dashboard(
    artifacts_dir: str | Path, output: str | Path
) -> tuple[Path, dict[str, object]]:
    """Write the self-contained dashboard to ``output``; returns (path, payloads)."""
    html, payloads = render_static_dashboard(artifacts_dir)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html)
    return out, payloads
