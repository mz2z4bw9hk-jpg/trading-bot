"""Render the dashboard in a real browser and capture screenshots.

Boots uvicorn over an artifacts directory, loads the page in headless
Chromium (Playwright), waits for the charts to draw, and captures full-page
screenshots in dark and light themes. Used as the visual verification step —
API tests prove the endpoints; this proves a human actually sees a dashboard.

Usage:
    python scripts/screenshot_dashboard.py [artifacts_dir] [out_dir]
"""

from __future__ import annotations

import contextlib
import os
import socket
import sys
import threading
import time
from pathlib import Path


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def main(artifacts_dir: str = "artifacts", out_dir: str = "artifacts") -> int:
    import uvicorn
    from playwright.sync_api import sync_playwright

    from titan.server.app import create_app

    artifacts = Path(artifacts_dir)
    if not (artifacts / "report.json").exists():
        print(f"no artifacts in {artifacts}; run `titan validate` first", file=sys.stderr)
        return 2
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(artifacts), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started:
        if time.time() > deadline:
            print("uvicorn failed to start", file=sys.stderr)
            return 1
        time.sleep(0.1)

    url = f"http://127.0.0.1:{port}/"
    exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE") or "/opt/pw-browsers/chromium"
    launch_kwargs = {}
    if Path(exe).exists():
        launch_kwargs["executable_path"] = exe

    with sync_playwright() as pw:
        browser = pw.chromium.launch(**launch_kwargs)
        page = browser.new_page(viewport={"width": 1440, "height": 1000})
        errors: list[str] = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on(
            "console",
            lambda m: errors.append(m.text) if m.type == "error" else None,
        )
        page.goto(url, wait_until="networkidle")
        page.wait_for_timeout(1200)  # let SVG charts finish drawing

        for theme in ("dark", "light"):
            page.evaluate(
                f"localStorage.setItem('titan-theme','{theme}');"
                f"document.documentElement.dataset.theme='{theme}';"
            )
            page.evaluate("render()")
            page.wait_for_timeout(500)
            path = out / f"dashboard_{theme}.png"
            page.screenshot(path=str(path), full_page=True)
            print(f"saved {path}")

        n_svg = page.locator("svg").count()
        n_cards = page.locator("section.card").count()
        browser.close()

    with contextlib.suppress(Exception):
        server.should_exit = True

    print(f"charts rendered: {n_svg} svg elements across {n_cards} cards")
    if errors:
        print("PAGE ERRORS:", *errors[:10], sep="\n  ", file=sys.stderr)
        return 1
    if n_svg < 4:
        print(f"expected >=4 charts, saw {n_svg}", file=sys.stderr)
        return 1
    print("dashboard visual verification OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(
            sys.argv[1] if len(sys.argv) > 1 else "artifacts",
            sys.argv[2] if len(sys.argv) > 2 else "artifacts",
        )
    )
