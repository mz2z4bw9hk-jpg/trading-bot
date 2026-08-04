"""Live last-price quotes, for marking an open book between bars.

The research pipeline reads *bars*. A daily bar changes once a day, so nothing
downstream of it can move faster than that — but an open position's value can,
and the paper account is worth watching between closes. This module supplies
the one number that changes: the latest traded price per symbol.

WHY THIS IS NOT THE PROVIDER. :class:`~titan.data.providers.YahooProvider`
fetches history per symbol, sized for research: hundreds of bars, one request
each, cached to disk. Marking a book needs the opposite — one number per
symbol, all symbols at once, repeatedly, never cached. Running the research
provider on a refresh loop would issue one request per symbol per tick; at 180
symbols that is 180 requests a minute, which is how an IP gets blocked.

RATE LIMITS ARE A CORRECTNESS PROBLEM, not an etiquette one. A blocked client
does not get slower data, it gets *no* data, and a dashboard that silently
falls back to stale prices while claiming to be live is worse than one that
never claimed it. So: every fetch is a single batched request, a minimum
interval is enforced regardless of what the caller asks for, failures back off
exponentially, and the age of the last successful quote travels with the
result so the UI can say how stale it is.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from titan.core.log import get_logger

logger = get_logger(__name__)

# Floor on how often a live vendor may be polled, whatever the caller requests.
# Yahoo's undocumented limits bite somewhere above a few requests a second from
# one IP; batching keeps a tick to ONE request, and this keeps ticks apart.
MIN_VENDOR_INTERVAL_SECONDS = 5.0

# Backoff after a failed fetch is base * 2^failures, capped. The base is
# independent of the poll interval so a zero-interval source still backs off.
BACKOFF_BASE_SECONDS = 1.0
MAX_BACKOFF_SECONDS = 300.0


@dataclass(frozen=True, slots=True)
class Quote:
    symbol: str
    price: float
    as_of: datetime

    @property
    def age_seconds(self) -> float:
        return max((datetime.now(UTC) - self.as_of).total_seconds(), 0.0)


class QuoteSource(Protocol):
    """Anything that can return a last price per symbol."""

    name: str

    def fetch(self, symbols: Iterable[str]) -> dict[str, Quote]:
        ...


class FrameQuotes:
    """Last close from already-loaded bars.

    The correct source for synthetic and CSV runs, where there is no live
    market to poll, and the fallback for a live run whose vendor is
    unreachable. It never fails and never goes to the network; the price it
    returns is simply as fresh as the bar it came from, which the caller can
    see from ``as_of``.
    """

    name = "frames"

    def __init__(self, frames: Mapping[str, Any]) -> None:
        self._frames = frames

    def fetch(self, symbols: Iterable[str]) -> dict[str, Quote]:
        out: dict[str, Quote] = {}
        for symbol in symbols:
            frame = self._frames.get(symbol)
            if frame is None or len(frame) == 0:
                continue
            try:
                price = float(frame["close"].iloc[-1])
                stamp = frame.index[-1].to_pydatetime()
            except (KeyError, IndexError, AttributeError, TypeError, ValueError):
                continue
            if price > 0:
                out[symbol] = Quote(symbol, price, stamp)
        return out


class YahooQuotes:
    """Batched last prices from Yahoo — one request for the whole universe.

    ``yfinance.download`` accepts a symbol list and issues a single chunked
    request, which is the only reason polling a 200-name book is viable at all.
    A 1-minute interval over the last day gives the freshest print Yahoo
    exposes without a paid feed; the last non-NaN close per column is the mark.
    """

    name = "yahoo"

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("live quotes require: pip install 'titan[data]'") from exc

    def fetch(self, symbols: Iterable[str]) -> dict[str, Quote]:  # pragma: no cover
        import pandas as pd
        import yfinance as yf

        wanted = list(dict.fromkeys(symbols))
        if not wanted:
            return {}
        data = yf.download(
            tickers=wanted,
            period="1d",
            interval="1m",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
        if data is None or len(data) == 0:
            raise RuntimeError("Yahoo returned no quote data")

        closes = data.get("Close", data)
        if isinstance(closes, pd.Series):          # single symbol: no column axis
            closes = closes.to_frame(name=wanted[0])

        out: dict[str, Quote] = {}
        for symbol in wanted:
            if symbol not in closes.columns:
                continue
            series = closes[symbol].dropna()
            if series.empty:
                continue
            price = float(series.iloc[-1])
            if price <= 0:
                continue
            stamp = series.index[-1]
            stamp = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
            if stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=UTC)
            out[symbol] = Quote(symbol, price, stamp)
        return out


def build_quote_source(cfg: Any, frames: Mapping[str, Any] | None = None) -> QuoteSource:
    """The right quote source for a config, falling back to bars.

    Only the yahoo provider has a live market behind it. Synthetic and CSV runs
    get :class:`FrameQuotes`, which is not a degraded mode — it is the honest
    answer when the "market" is a file.
    """
    if getattr(cfg.data, "provider", None) == "yahoo":
        try:
            return YahooQuotes()
        except ImportError as exc:
            logger.warning("live quotes unavailable (%s); marking from bars instead", exc)
    return FrameQuotes(frames or {})


class QuoteCache:
    """Polls a source on an interval and hands out the freshest prices it has.

    Decoupling *poll rate* from *read rate* is the point. A dashboard can read
    this every second — it is a dict lookup — while the vendor behind it is
    touched at whatever cadence is safe. Readers never block on the network and
    never see a half-updated map.

    Failures degrade rather than propagate: the last good prices stay served,
    the error is recorded for display, and the next attempt is pushed out
    exponentially so a vendor outage does not become a request flood.
    """

    def __init__(
        self,
        source: QuoteSource,
        symbols: Iterable[str],
        *,
        interval_seconds: float = 15.0,
        min_interval_seconds: float | None = None,
    ) -> None:
        self._source = source
        self._symbols = list(dict.fromkeys(symbols))
        floor = (
            MIN_VENDOR_INTERVAL_SECONDS
            if min_interval_seconds is None and source.name != "frames"
            else (min_interval_seconds or 0.0)
        )
        self._interval = max(float(interval_seconds), floor)
        if self._interval > interval_seconds:
            logger.info(
                "quote interval raised from %.1fs to %.1fs: %s is a live vendor and "
                "polling it faster risks being rate-limited into no data at all",
                interval_seconds, self._interval, source.name,
            )
        self._lock = threading.Lock()
        self._quotes: dict[str, Quote] = {}
        self._last_attempt = 0.0
        self._last_success: float | None = None
        self._failures = 0
        self._error: str | None = None

    @property
    def interval(self) -> float:
        return self._interval

    def _backoff(self) -> float:
        """Extra delay after failures, on its own base rather than the interval.

        Deriving it from the poll interval alone breaks exactly where it
        matters: a bar-backed source runs at interval 0, so ``interval * 2^n``
        stays 0 and a failing source gets retried in a tight loop — the
        opposite of backing off.
        """
        if not self._failures:
            return 0.0
        base = max(self._interval, BACKOFF_BASE_SECONDS)
        return min(base * (2 ** self._failures), MAX_BACKOFF_SECONDS)

    def due(self) -> bool:
        now = time.monotonic()
        return now - self._last_attempt >= self._interval + self._backoff()

    def refresh(self, force: bool = False) -> bool:
        """Poll if due. Returns True when new prices were stored."""
        if not force and not self.due():
            return False
        self._last_attempt = time.monotonic()
        try:
            fetched = self._source.fetch(self._symbols)
        except Exception as exc:  # vendor errors are wide and not worth enumerating
            self._failures += 1
            self._error = f"{type(exc).__name__}: {exc}"
            logger.warning(
                "quote fetch failed (%d in a row, next attempt in %.0fs): %s",
                self._failures, self._interval + self._backoff(), self._error,
            )
            return False
        if not fetched:
            self._failures += 1
            self._error = "vendor returned no quotes"
            return False
        with self._lock:
            self._quotes.update(fetched)
        self._failures = 0
        self._error = None
        self._last_success = time.monotonic()
        return True

    def prices(self) -> dict[str, float]:
        with self._lock:
            return {s: q.price for s, q in self._quotes.items()}

    def status(self) -> dict[str, Any]:
        with self._lock:
            quotes = list(self._quotes.values())
        newest = max((q.as_of for q in quotes), default=None)
        return {
            "source": self._source.name,
            "n_quotes": len(quotes),
            "interval_seconds": round(self._interval, 1),
            "quote_time": newest.isoformat() if newest else None,
            "quote_age_seconds": (
                round(max((datetime.now(UTC) - newest).total_seconds(), 0.0), 1)
                if newest else None
            ),
            "seconds_since_success": (
                round(time.monotonic() - self._last_success, 1)
                if self._last_success is not None else None
            ),
            "consecutive_failures": self._failures,
            "error": self._error,
        }
