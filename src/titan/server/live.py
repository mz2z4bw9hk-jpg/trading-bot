"""Live account refresh: keep ``account.json`` current while the server runs.

Without this the dashboard polls files that only change when someone runs a
command. The page ticks, the numbers do not, and "live" means nothing.

WHAT ACTUALLY CHANGES, AND HOW OFTEN. Three clocks, deliberately separate,
because collapsing them either wastes requests or reports stale data as fresh:

- **Prices** move continuously, and are fetched in one batched request per
  poll (:mod:`titan.data.quotes`). This is the only clock bounded by an outside
  party, so it is the only one with a floor on how fast it may run.
- **The account** is pure arithmetic over the tracking log and the newest
  prices. No network, so it can be recomputed as fast as anyone wants to look.
- **Bars** arrive on the config's timeframe — once a day on a ``1d`` config.
  Reloading them on a fast loop would re-download history to learn nothing, so
  they refresh on a long interval and whenever the calendar date rolls over.

WHAT THIS DOES NOT DO. It does not re-run the scanner. A new order requires a
new *bar* — the model gate, the technical setups and the labels are all defined
on closes — so inventing intra-bar signals would be fabricating research the
pipeline never did. New orders come from ``titan scan`` when a bar closes. What
moves in between is the valuation of what is already open, which is exactly
what a broker screen shows between fills.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from titan.core.config import TitanConfig
from titan.core.jsonsafe import json_safe
from titan.core.log import get_logger

logger = get_logger(__name__)

# The page may poll faster, but recomputing below this is spending CPU to
# redraw identical numbers — quotes cannot move faster than their own floor.
MIN_ACCOUNT_INTERVAL_SECONDS = 0.5

# How often to re-read bars. A daily config gains one bar a day; this exists so
# a long-running server eventually notices, not so it stays current by polling.
DEFAULT_FRAME_RELOAD_SECONDS = 900.0


class LiveAccount:
    """Background thread that rewrites ``account.json`` from live prices.

    Errors never kill the loop and never blank the file: a failed tick leaves
    the last good artifact in place and records why, so the dashboard can show
    a stale-but-honest number rather than an empty panel.
    """

    def __init__(
        self,
        cfg: TitanConfig,
        artifacts_dir: str | Path,
        *,
        interval_seconds: float = 1.0,
        quote_interval_seconds: float = 15.0,
        frame_reload_seconds: float = DEFAULT_FRAME_RELOAD_SECONDS,
    ) -> None:
        self._cfg = cfg
        self._artifacts = Path(artifacts_dir)
        self._interval = max(float(interval_seconds), MIN_ACCOUNT_INTERVAL_SECONDS)
        self._quote_interval = float(quote_interval_seconds)
        self._frame_reload = float(frame_reload_seconds)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()

        self._frames: dict[str, Any] = {}
        self._marks: dict[str, Any] = {}
        self._frames_loaded_at: float | None = None
        self._frames_date: str | None = None
        self._cache: Any = None

        self._ticks = 0
        self._last_tick: datetime | None = None
        self._last_error: str | None = None
        self._started_at: datetime | None = None

    # ------------------------------------------------------------------ #

    def start(self) -> None:
        if self._thread is not None:
            return
        self._started_at = datetime.now(UTC)
        self._thread = threading.Thread(
            target=self._run, name="titan-live-account", daemon=True
        )
        self._thread.start()
        logger.info(
            "live account: refreshing every %.1fs, quotes every %.1fs",
            self._interval, self._quote_interval,
        )

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    # ------------------------------------------------------------------ #

    def _load_frames(self) -> None:
        from titan.data.store import MarketDataStore
        from titan.monitor.account import daily_marks

        dataset = MarketDataStore(
            self._cfg.data, self._cfg.universe, seed=self._cfg.run.seed
        ).load()
        self._frames = dataset.frames
        # Aggregated once per reload, not once per tick. It is a groupby over
        # every bar of every symbol — ~900ms on a 24-name book — and it cannot
        # change until the bars do, so recomputing it per tick would spend the
        # entire refresh budget rediscovering the same numbers.
        self._marks = daily_marks(self._frames)
        self._frames_loaded_at = time.monotonic()
        self._frames_date = datetime.now(UTC).strftime("%Y-%m-%d")

    def _ensure_frames(self) -> None:
        """Reload bars on the long clock, or as soon as the date rolls over."""
        from titan.monitor.account import daily_marks

        today = datetime.now(UTC).strftime("%Y-%m-%d")
        stale = (
            self._frames_loaded_at is None
            or time.monotonic() - self._frames_loaded_at > self._frame_reload
            or self._frames_date != today
        )
        if stale:
            self._load_frames()
        elif self._frames and not self._marks:
            # Marks must always describe the frames beside them. Empty marks
            # with non-empty bars is not "nothing to mark" — it silently skips
            # marking entirely, which is the kind of failure that shows a
            # plausible number rather than an obviously broken one.
            self._marks = daily_marks(self._frames)

    def _quote_symbols(self, records: list[dict]) -> list[str]:
        """Only the symbols the account actually holds.

        Marking an open book needs a price for each OPEN POSITION — nothing
        else. Requesting the whole configured universe instead was the defect
        behind a live run that hammered the vendor into refusing service: 201
        symbols asked for, 9 needed, every second. The universe is what the
        scanner ranks once a bar; the open book is what a mark re-prices, and
        it is typically a dozen names.

        Read straight off the log rather than from a replay: an unresolved
        record is an open position, and that is a scan of a small JSON file
        rather than a full ledger rebuild on every tick.
        """
        held = {
            str(r["symbol"]) for r in records
            if r.get("outcome") is None and r.get("size_fraction") is not None
        }
        return sorted(held)

    def _ensure_cache(self, symbols: list[str]) -> None:
        """Build or rebuild the quote cache for the currently held symbols.

        Rebuilt when the open book changes — a scan that opens a position adds
        a symbol that needs pricing, and one that closes removes a symbol there
        is no longer any reason to ask about.
        """
        from titan.data.quotes import QuoteCache, build_quote_source

        stale_source = (
            self._cache is not None
            and getattr(self._cache, "_source", None).__class__.__name__ == "FrameQuotes"
            # The bar-backed source holds a reference to the frames dict it was
            # built with; a reload replaces that dict, so rebuild against the
            # new one rather than pricing from a frame set nothing else reads.
            and getattr(self._cache._source, "_frames", None) is not self._frames
        )
        if self._cache is None or stale_source or self._cache._symbols != symbols:
            self._cache = QuoteCache(
                build_quote_source(self._cfg, self._frames),
                symbols,
                interval_seconds=self._quote_interval,
            )

    def tick(self) -> dict[str, Any]:
        """One refresh: newest bars, newest prices, recomputed account."""
        from titan.monitor.account import replay
        from titan.monitor.paper import PaperTrackingStore, paper_store_path

        self._ensure_frames()
        store = PaperTrackingStore(paper_store_path(self._cfg))
        records = store.records
        self._ensure_cache(self._quote_symbols(records))
        self._cache.refresh()
        status = self._cache.status()
        state = replay(
            records,
            starting_equity=self._cfg.monitor.paper_starting_equity,
            max_gross_exposure=self._cfg.backtest.max_gross_exposure,
            max_account_leverage=self._cfg.risk.leverage.max_account_leverage,
            frames=self._frames,
            marks=self._marks,
            quotes=self._cache.prices(),
            quote_time=status.get("quote_time"),
        )
        state["live"] = self.status(quotes=status)

        self._artifacts.mkdir(parents=True, exist_ok=True)
        # Write-then-rename: a reader polling every second must never catch a
        # half-written file and render a truncated account.
        tmp = self._artifacts / "account.json.tmp"
        tmp.write_text(json.dumps(json_safe(state), indent=1, default=str, allow_nan=False))
        tmp.replace(self._artifacts / "account.json")
        return state

    def _run(self) -> None:
        while not self._stop.is_set():
            began = time.monotonic()
            try:
                self.tick()
                self._ticks += 1
                self._last_tick = datetime.now(UTC)
                self._last_error = None
            except Exception as exc:
                self._last_error = f"{type(exc).__name__}: {exc}"
                logger.warning("live account tick failed: %s", self._last_error)
            # Sleep the remainder, so a slow tick does not compound into drift.
            self._stop.wait(max(self._interval - (time.monotonic() - began), 0.05))

    # ------------------------------------------------------------------ #

    def status(self, quotes: dict[str, Any] | None = None) -> dict[str, Any]:
        if quotes is None:
            quotes = self._cache.status() if self._cache is not None else {}
        return {
            "enabled": True,
            "refresh_seconds": round(self._interval, 2),
            "ticks": self._ticks,
            "last_tick": self._last_tick.isoformat() if self._last_tick else None,
            "started_at": self._started_at.isoformat() if self._started_at else None,
            "error": self._last_error,
            "bars_loaded_at": self._frames_date,
            "n_symbols": len(self._frames),
            "quotes": quotes,
        }
