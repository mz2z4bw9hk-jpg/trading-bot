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

import logging
import threading
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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


@contextmanager
def _quiet(*loggers: str) -> Iterator[None]:
    """Suppress a third-party library's own logging for the duration.

    yfinance logs one ERROR line per symbol it could not fetch, plus a summary
    listing them all. On a refresh loop over a 178-name universe that is
    thousands of lines a minute of someone else's error reporting, drowning
    every message this platform emits. The failures are not ignored — they come
    back as a coverage number in :meth:`QuoteCache.status` — they are just not
    reprinted verbatim on every tick.
    """
    saved = [(logging.getLogger(n), logging.getLogger(n).level) for n in loggers]
    for log, _ in saved:
        log.setLevel(logging.CRITICAL)
    try:
        yield
    finally:
        for log, level in saved:
            log.setLevel(level)


class YahooQuotes:
    """Batched last prices from Yahoo.

    Three things here are load-bearing, each learned from a failure:

    **Daily bars, not 1-minute.** The current day's daily bar is *in progress*
    during a session: its close IS the last trade. Asking for 1m bars instead
    buys nothing for marking a book, costs far more, and is the request Yahoo
    rejects first — a 178-symbol 1m call comes back as "possibly delisted" for
    most of the universe.

    **Chunked, and single-threaded.** ``yf.download(threads=True)`` fans out
    across worker threads that share a sqlite timezone cache; called from a
    background refresh thread, that races and every symbol fails with
    ``OperationalError: unable to open database file``. Sequential chunks are
    slower and actually work, which at a 15-second interval is the right trade.

    **Coverage is reported, not assumed.** Yahoo answers a batch partially all
    the time. Returning "some prices" as though it were success hides a feed
    that is 40% blind, so the caller is told how many of the symbols it asked
    for came back.
    """

    name = "yahoo"

    # Yahoo degrades sharply with batch size; 50 is comfortably inside where
    # partial failures start.
    CHUNK = 50

    # Ceiling on the per-symbol fallback. An open book is a dozen names, so
    # this never binds in normal use; it exists so that a caller who does pass
    # a whole universe cannot turn one failed batch into 200 serial requests.
    MAX_FALLBACK = 25

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ImportError("live quotes require: pip install 'titan[data]'") from exc
        # Point the timezone cache somewhere writable and stable. Left at its
        # default it lands in a platform dir that may not exist, and every
        # lookup fails with a sqlite error that surfaces as "delisted".
        if cache_dir is not None:
            try:
                path = Path(cache_dir) / "yf_tz_cache"
                path.mkdir(parents=True, exist_ok=True)
                yf.set_tz_cache_location(str(path))
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug("could not set yfinance tz cache location: %s", exc)

    def fetch(self, symbols: Iterable[str]) -> dict[str, Quote]:  # pragma: no cover
        import pandas as pd
        import yfinance as yf

        wanted = list(dict.fromkeys(symbols))
        if not wanted:
            return {}

        out: dict[str, Quote] = {}
        errors: list[str] = []
        with _quiet("yfinance", "yfinance.data", "peewee"):
            for i in range(0, len(wanted), self.CHUNK):
                chunk = wanted[i : i + self.CHUNK]
                try:
                    data = yf.download(
                        tickers=chunk,
                        period="5d",          # today's in-progress bar, plus slack
                        interval="1d",
                        auto_adjust=True,
                        progress=False,
                        group_by="column",
                        threads=False,        # see class docstring
                    )
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                if data is None or len(data) == 0:
                    continue

                closes = data.get("Close", data)
                if isinstance(closes, pd.Series):   # single symbol: no column axis
                    closes = closes.to_frame(name=chunk[0])
                for symbol in chunk:
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

        # Per-symbol fallback for whatever the batch did not answer. This is
        # the exact call the research provider makes — proven to work wherever
        # the dataset itself loads — and it is affordable here only because the
        # caller asks for the OPEN BOOK, a dozen names, not the universe.
        missing = [s for s in wanted if s not in out]
        if missing and len(missing) <= self.MAX_FALLBACK:
            with _quiet("yfinance", "yfinance.data", "peewee"):
                for symbol in missing:
                    try:
                        df = yf.Ticker(symbol).history(
                            period="5d", interval="1d", auto_adjust=True
                        )
                    except Exception:  # one bad symbol must not abort the rest
                        continue
                    if df is None or df.empty or "Close" not in df:
                        continue
                    series = df["Close"].dropna()
                    if series.empty or float(series.iloc[-1]) <= 0:
                        continue
                    stamp = series.index[-1]
                    stamp = stamp.to_pydatetime() if hasattr(stamp, "to_pydatetime") else stamp
                    if stamp.tzinfo is None:
                        stamp = stamp.replace(tzinfo=UTC)
                    out[symbol] = Quote(symbol, float(series.iloc[-1]), stamp)

        if not out and errors:
            raise RuntimeError(f"every quote chunk failed: {errors[0]}")
        return out


def build_quote_source(cfg: Any, frames: Mapping[str, Any] | None = None) -> QuoteSource:
    """The right quote source for a config, falling back to bars.

    Only the yahoo provider has a live market behind it. Synthetic and CSV runs
    get :class:`FrameQuotes`, which is not a degraded mode — it is the honest
    answer when the "market" is a file.
    """
    if getattr(cfg.data, "provider", None) == "yahoo":
        try:
            return YahooQuotes(cache_dir=getattr(cfg.data, "cache_dir", None))
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
        min_coverage: float = 0.5,
    ) -> None:
        self._source = source
        self._symbols = list(dict.fromkeys(symbols))
        self._min_coverage = min_coverage
        self._coverage = 0.0
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
        """Poll if due. Returns True when new prices were stored.

        A *partial* answer is not treated as success. Yahoo routinely returns
        half a batch, and counting that as healthy resets the backoff — so a
        feed that is chronically 60% blind gets polled at full rate forever,
        which is both useless and the surest way to stay rate-limited. Below
        ``min_coverage`` the prices are still kept (they are real) but the
        attempt counts as a failure so the interval stretches.
        """
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

        if fetched:
            with self._lock:
                self._quotes.update(fetched)
        coverage = len(fetched) / len(self._symbols) if self._symbols else 0.0
        self._coverage = coverage

        if coverage < self._min_coverage:
            self._failures += 1
            self._error = (
                f"only {len(fetched)}/{len(self._symbols)} symbols returned a price "
                f"({coverage:.0%} coverage)"
            )
            logger.warning(
                "quote fetch degraded (%d in a row, next attempt in %.0fs): %s",
                self._failures, self._interval + self._backoff(), self._error,
            )
            return bool(fetched)

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
            "n_symbols": len(self._symbols),
            "coverage": round(self._coverage, 3),
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
