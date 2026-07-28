"""The scanner must not drop instruments for not trading on the newest day.

Scanning one global date silently loses every instrument whose calendar does
not include it. On a mixed equity + crypto universe that is the whole equity
book, every weekend: crypto prints a Saturday bar, the panel's newest date
becomes Saturday, and ~100 stocks vanish from the scan with no message. The
observed symptom was a 178-instrument run reporting "scan: 77 instruments".

Each instrument is therefore scanned on its own freshest bar, with a staleness
ceiling so a delisted symbol cannot signal from months-old prices.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from titan.core.config import TitanConfig
from titan.features.registry import FeaturePanel


class _StubEnsemble:
    """Predicts a constant probability; the calendar is what is under test."""

    has_intervals = False

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.4), np.full(len(X), 0.6)])

    def uncertainty(self, X):
        return np.full(len(X), 0.01)


class _StubDetector:
    def snapshot(self, frame):
        from titan.core.types import Regime, VolState

        class _S:
            regime = Regime.BULL
            vol_state = VolState.NORMAL
            confidence = 0.9

        return _S()


class _StubGenerator:
    """Never emits: rows and their statuses are what matters here."""

    def generate(self, **kwargs):
        return None


def _mixed_calendar_panel() -> tuple[FeaturePanel, dict]:
    """Two equities on business days, two coins on every day — crypto fresher."""
    equity_dates = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60), tz="UTC")
    # Crypto covers the same span and keeps printing for two more days — the
    # weekend case that made the newest panel date crypto-only.
    crypto_dates = pd.DatetimeIndex(
        pd.date_range(
            equity_dates[0], equity_dates[-1] + pd.Timedelta(days=2), freq="D", tz="UTC"
        )
    )

    rng = np.random.default_rng(0)
    rows, index = [], []
    frames = {}
    for sym, dates in [
        ("AAPL", equity_dates), ("MSFT", equity_dates),
        ("BTC-USD", crypto_dates), ("ETH-USD", crypto_dates),
    ]:
        for d in dates:
            rows.append(rng.normal(size=3))
            index.append((d, sym))
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(dates))))
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(len(dates), 1e6)},
            index=dates,
        )

    X = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["date", "symbol"]),
        columns=["f0", "f1", "f2"],
    ).sort_index()
    return FeaturePanel(X=X), frames


class _StubDataset:
    def __init__(self, frames):
        self.frames = frames
        self.reliability = dict.fromkeys(frames, 1.0)
        longest = max(frames.values(), key=len)
        self.benchmark_frame = longest


def _scan(max_staleness_bars: int = 5):
    from titan.scanner.scanner import MarketScanner

    panel, frames = _mixed_calendar_panel()
    cfg = TitanConfig()
    cfg.scanner.max_staleness_bars = max_staleness_bars
    scanner = MarketScanner(
        cfg,
        ensemble=_StubEnsemble(),
        selected_features=["f0", "f1", "f2"],
        generator=_StubGenerator(),
        detector=_StubDetector(),
    )
    return scanner.scan(_StubDataset(frames), panel)


def test_every_instrument_appears_in_the_scan():
    """The reported failure: 178 instruments in, 77 out."""
    result = _scan()
    assert {r.symbol for r in result.rows} == {"AAPL", "MSFT", "BTC-USD", "ETH-USD"}


def test_equities_are_ranked_not_dropped_when_crypto_has_a_fresher_bar():
    result = _scan()
    equities = [r for r in result.rows if r.symbol in {"AAPL", "MSFT"}]

    assert len(equities) == 2
    for row in equities:
        assert "stale" not in row.status
        assert np.isfinite(row.probability)


def test_a_genuinely_stale_instrument_is_reported_not_ranked():
    """Zero tolerance: everything not on the newest date must be flagged."""
    result = _scan(max_staleness_bars=0)

    stale = [r for r in result.rows if r.status.startswith("stale")]
    assert {r.symbol for r in stale} == {"AAPL", "MSFT"}
    assert all("last bar" in r.status for r in stale)
    # Still present in the output — reported, not silently dropped.
    assert len(result.rows) == 4


def test_scan_date_remains_the_newest_panel_date():
    result = _scan()
    assert result.date == max(
        max(f.index) for f in _mixed_calendar_panel()[1].values()
    )


def test_every_row_carries_a_status():
    for row in _scan().rows:
        assert row.status
