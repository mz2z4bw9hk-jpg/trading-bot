"""Live quotes: refresh rate, failure behaviour, and not getting blocked.

The thing that breaks a polling client is not a wrong price, it is a vendor
that stops answering. So the properties under test are mostly about restraint:
a floor on how fast a live source may be polled whatever the caller asked for,
backoff after failures, and — most importantly — that a failed fetch keeps
serving the last good prices instead of blanking the book.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.data.quotes import (
    MIN_VENDOR_INTERVAL_SECONDS,
    FrameQuotes,
    Quote,
    QuoteCache,
    build_quote_source,
)


def _frame(closes, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.DatetimeIndex(pd.bdate_range(start, periods=len(closes)), tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes, "low": closes, "close": closes,
         "volume": np.full(len(closes), 1e6)},
        index=idx,
    )


class _Stub:
    """A source under the test's control: prices, failures, call count."""

    name = "stub"

    def __init__(self, prices=None, fail_with=None):
        self.prices = {"AAA": 100.0} if prices is None else prices
        self.fail_with = fail_with
        self.calls = 0

    def fetch(self, symbols):
        self.calls += 1
        if self.fail_with:
            raise self.fail_with
        return {
            s: Quote(s, p, pd.Timestamp.now(tz="UTC").to_pydatetime())
            for s, p in self.prices.items()
        }


# ------------------------------------------------------------ frames -------


def test_frame_quotes_read_the_last_close():
    src = FrameQuotes({"AAA": _frame([10.0, 11.0, 12.5])})
    quotes = src.fetch(["AAA"])

    assert quotes["AAA"].price == pytest.approx(12.5)
    assert quotes["AAA"].as_of.year == 2024


def test_frame_quotes_skip_symbols_they_do_not_have():
    src = FrameQuotes({"AAA": _frame([10.0])})
    assert src.fetch(["AAA", "MISSING"]).keys() == {"AAA"}


def test_frame_quotes_survive_an_empty_frame():
    src = FrameQuotes({"AAA": _frame([10.0]).iloc[:0], "BBB": _frame([5.0])})
    assert set(src.fetch(["AAA", "BBB"])) == {"BBB"}


def test_a_non_yahoo_config_gets_frame_quotes():
    """Synthetic and CSV runs have no live market. That is not a degraded mode."""
    from titan.core.config import TitanConfig

    cfg = TitanConfig()
    assert cfg.data.provider == "synthetic"
    assert build_quote_source(cfg, {"AAA": _frame([1.0])}).name == "frames"


# ------------------------------------------------------------- cache -------


def test_the_cache_serves_prices_without_touching_the_source():
    """Read rate is decoupled from poll rate — the whole point of the cache."""
    src = _Stub({"AAA": 42.0})
    cache = QuoteCache(src, ["AAA"], interval_seconds=60, min_interval_seconds=0)
    cache.refresh(force=True)

    for _ in range(100):
        assert cache.prices() == {"AAA": 42.0}
    assert src.calls == 1


def test_refresh_is_a_no_op_until_the_interval_elapses():
    src = _Stub()
    cache = QuoteCache(src, ["AAA"], interval_seconds=3600, min_interval_seconds=0)

    assert cache.refresh(force=True) is True
    assert cache.refresh() is False          # not due
    assert src.calls == 1


def test_a_live_vendor_cannot_be_polled_faster_than_the_floor():
    """The caller asking for 0.1s does not make 0.1s safe.

    Being rate-limited does not yield slower data, it yields none — so the
    floor is a correctness guard, not politeness.
    """
    src = _Stub()
    src.name = "yahoo"
    cache = QuoteCache(src, ["AAA"], interval_seconds=0.1)

    assert cache.interval == MIN_VENDOR_INTERVAL_SECONDS


def test_a_bar_backed_source_has_no_floor():
    """No vendor, no rate limit: reading a DataFrame cannot be throttled."""
    cache = QuoteCache(FrameQuotes({}), ["AAA"], interval_seconds=0.1)
    assert cache.interval == pytest.approx(0.1)


def test_a_failed_fetch_keeps_serving_the_last_good_prices():
    """A dropped poll must not empty the book."""
    src = _Stub({"AAA": 42.0})
    cache = QuoteCache(src, ["AAA"], interval_seconds=0, min_interval_seconds=0)
    cache.refresh(force=True)

    src.fail_with = RuntimeError("vendor down")
    assert cache.refresh(force=True) is False
    assert cache.prices() == {"AAA": 42.0}


def test_failures_are_reported_rather_than_raised():
    src = _Stub(fail_with=RuntimeError("429 Too Many Requests"))
    cache = QuoteCache(src, ["AAA"], interval_seconds=0, min_interval_seconds=0)
    cache.refresh(force=True)
    status = cache.status()

    assert status["consecutive_failures"] == 1
    assert "429" in status["error"]
    assert status["n_quotes"] == 0


def test_repeated_failures_back_off_instead_of_hammering():
    src = _Stub(fail_with=RuntimeError("down"))
    cache = QuoteCache(src, ["AAA"], interval_seconds=1.0, min_interval_seconds=0)

    cache.refresh(force=True)
    first = cache._backoff()
    cache.refresh(force=True)
    second = cache._backoff()

    assert second > first > 0
    assert not cache.due()          # the next attempt is pushed out


def test_a_success_clears_the_backoff():
    src = _Stub(fail_with=RuntimeError("down"))
    cache = QuoteCache(src, ["AAA"], interval_seconds=0, min_interval_seconds=0)
    cache.refresh(force=True)
    assert cache._backoff() > 0

    src.fail_with = None
    cache.refresh(force=True)
    assert cache._backoff() == 0
    assert cache.status()["error"] is None


def test_an_empty_vendor_response_counts_as_a_failure():
    """Zero quotes is not success — it would silently freeze every mark."""
    src = _Stub(prices={})
    cache = QuoteCache(src, ["AAA"], interval_seconds=0, min_interval_seconds=0)

    assert cache.refresh(force=True) is False
    assert cache.status()["consecutive_failures"] == 1


def test_status_reports_quote_age_for_the_ui():
    src = _Stub({"AAA": 1.0})
    cache = QuoteCache(src, ["AAA"], interval_seconds=0, min_interval_seconds=0)
    cache.refresh(force=True)
    status = cache.status()

    assert status["n_quotes"] == 1
    assert status["quote_age_seconds"] is not None
    assert status["quote_age_seconds"] < 5


def test_duplicate_symbols_are_requested_once():
    src = _Stub()
    cache = QuoteCache(src, ["AAA", "AAA", "BBB", "AAA"], min_interval_seconds=0)
    assert cache._symbols == ["AAA", "BBB"]
