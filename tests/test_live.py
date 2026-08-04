"""The live refresh loop: what it updates, and what it must not.

The valuable property here is a negative one. A refresh may re-price an open
book as fast as anyone wants, but it must never manufacture an *order* — the
gate, the setups and the labels are all defined on bar closes, so an
intra-bar signal would be research the pipeline never did. The rest is
plumbing that has to survive being run once a second forever: no partial
writes, no unhandled exception killing the thread, no unbounded growth.
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from titan.core.config import TitanConfig
from titan.monitor.paper import paper_store_path
from titan.server.live import MIN_ACCOUNT_INTERVAL_SECONDS, LiveAccount


def _frame(closes, start="2024-01-02"):
    closes = np.asarray(closes, dtype=float)
    idx = pd.DatetimeIndex(pd.bdate_range(start, periods=len(closes)), tz="UTC")
    return pd.DataFrame(
        {"open": closes, "high": closes * 1.001, "low": closes * 0.999,
         "close": closes, "volume": np.full(len(closes), 1e6)},
        index=idx,
    )


def _record(symbol="AAA", date="2024-01-02", size=0.10, entry=100.0):
    return {
        "symbol": symbol, "date": date, "exit_date": None, "outcome": None,
        "touch": None, "ret": None, "bars_held": None,
        "side": "long", "source": "model", "asset_class": "equity",
        "size_fraction": size, "risk_percentage": 0.5, "cost_estimate": 0.0,
        "signal_entry": entry, "entry_price": entry,
        "stop_loss": entry * 0.95, "take_profit_levels": [entry * 1.1],
        "leverage": 1.0, "margin_fraction": size, "liquidation_price": None,
    }


@pytest.fixture
def wired(tmp_path):
    """A config whose store and artifacts live in tmp, with one open position."""
    cfg = TitanConfig()
    cfg.model.store_dir = tmp_path / "models"
    cfg.model.store_dir.mkdir(parents=True)
    paper_store_path(cfg).write_text(json.dumps({"records": [_record()]}))

    live = LiveAccount(cfg, tmp_path / "artifacts", interval_seconds=1, quote_interval_seconds=0)
    # Pre-seed the bars so no data store is loaded: these tests are about the
    # refresh loop, not about fetching. Marked as loaded NOW so the staleness
    # check leaves them alone.
    live._frames = {"AAA": _frame([100.0, 105.0, 110.0])}
    live._frames_loaded_at = time.monotonic()
    live._frames_date = datetime.now(UTC).strftime("%Y-%m-%d")
    return cfg, live, tmp_path


# --------------------------------------------------------------- ticks ------


def test_a_tick_writes_a_marked_account(wired):
    _, live, tmp = wired
    state = live.tick()

    assert state["n_open"] == 1
    assert state["account_value"] > state["starting_equity"]   # +10% and open
    assert (tmp / "artifacts" / "account.json").exists()


def test_the_written_file_is_valid_json_with_no_nan(wired):
    """FastAPI serializes with allow_nan=False; a NaN here 500s the endpoint."""
    _, live, tmp = wired
    live.tick()
    text = (tmp / "artifacts" / "account.json").read_text()

    payload = json.loads(text)
    assert "NaN" not in text and "Infinity" not in text
    assert payload["live"]["enabled"] is True


def test_the_write_is_atomic_leaving_no_partial_file(wired):
    """A reader polling every second must never catch a half-written file."""
    _, live, tmp = wired
    live.tick()

    assert not (tmp / "artifacts" / "account.json.tmp").exists()


def test_repeated_ticks_do_not_grow_the_equity_curve(wired):
    """The live point is replaced, not appended, or an hour adds 3600 points."""
    _, live, _ = wired
    first = len(live.tick()["equity_curve"])
    for _ in range(5):
        live.tick()
    assert len(live.tick()["equity_curve"]) == first


def test_a_tick_never_creates_an_order(wired):
    """The property that matters: no new bar, no new signal.

    Re-pricing is arithmetic on what already exists. Emitting an order between
    closes would be inventing a decision the walk-forward never validated.
    """
    cfg, live, _ = wired
    before = json.loads(paper_store_path(cfg).read_text())["records"]
    for _ in range(3):
        live.tick()
    after = json.loads(paper_store_path(cfg).read_text())["records"]

    assert before == after


def test_ticks_track_a_moving_price(wired):
    _, live, _ = wired
    first = live.tick()["account_value"]

    live._frames = {"AAA": _frame([100.0, 105.0, 130.0])}
    live._cache = None                      # rebuilt against the new frames
    assert live.tick()["account_value"] > first


# -------------------------------------------------------------- status ------


def test_status_reports_the_poll_rate_and_quote_freshness(wired):
    _, live, _ = wired
    live.tick()
    status = live.status()

    assert status["enabled"] is True
    assert status["refresh_seconds"] == 1
    assert status["quotes"]["source"] == "frames"
    assert status["quotes"]["n_quotes"] >= 1


def test_the_account_interval_has_a_floor(wired):
    cfg, _, tmp = wired
    live = LiveAccount(cfg, tmp / "a", interval_seconds=0.001)
    assert live._interval == MIN_ACCOUNT_INTERVAL_SECONDS


def test_a_failing_tick_leaves_the_last_good_artifact_in_place(wired):
    """A refresh that dies must not blank the dashboard."""
    _, live, tmp = wired
    good = live.tick()["account_value"]

    live._ensure_frames = lambda: (_ for _ in ()).throw(RuntimeError("vendor down"))
    with pytest.raises(RuntimeError):
        live.tick()

    payload = json.loads((tmp / "artifacts" / "account.json").read_text())
    assert payload["account_value"] == good


def test_the_thread_survives_a_failing_tick(wired):
    """_run swallows tick errors; a transient failure must not end the loop."""
    _, live, _ = wired
    def boom():
        live._stop.set()                   # one pass, then exit the loop
        raise RuntimeError("boom")

    live._ensure_frames = boom
    live._run()                            # must not raise

    assert live._last_error is not None and "boom" in live._last_error


# ----------------------------------------------------------------- api ------


def test_the_api_reports_static_mode_when_nothing_is_refreshing(tmp_path):
    from fastapi.testclient import TestClient

    from titan.server.app import create_app

    client = TestClient(create_app(tmp_path))
    payload = client.get("/api/live").json()

    assert payload["enabled"] is False
    assert payload["refresh_seconds"] == 30      # no reason to poll fast
    assert "titan scan" in payload["reason"]


def test_the_api_reports_the_live_interval_when_refreshing(wired):
    from fastapi.testclient import TestClient

    from titan.server.app import create_app

    _, live, tmp = wired
    live.tick()
    client = TestClient(create_app(tmp / "artifacts", live=live))
    payload = client.get("/api/live").json()

    assert payload["enabled"] is True
    assert payload["refresh_seconds"] == 1
    assert payload["quotes"]["source"] == "frames"


def test_the_account_endpoint_serves_the_refreshed_file(wired):
    from fastapi.testclient import TestClient

    from titan.server.app import create_app

    _, live, tmp = wired
    state = live.tick()
    client = TestClient(create_app(tmp / "artifacts", live=live))

    served = client.get("/api/account").json()
    assert served["account_value"] == state["account_value"]
    assert served["marked_live"] is True


# ------------------------------------------------------------ tick cost -----


def test_marks_are_computed_once_per_reload_not_once_per_tick(wired, monkeypatch):
    """The fix for a loop that could not keep up with its own interval.

    Aggregating every bar of every symbol is the expensive half of a replay and
    cannot change until the bars do. Recomputing it per tick measured ~900ms on
    a 24-name book — the entire budget of a 1s refresh, spent rediscovering
    identical numbers.
    """
    import titan.monitor.account as account_mod

    calls = []
    real = account_mod.daily_marks

    def counted(frames):
        calls.append(1)
        return real(frames)

    monkeypatch.setattr(account_mod, "daily_marks", counted)

    _, live, _ = wired
    live._marks = {}                       # as a fresh load would leave it
    for _ in range(10):
        live.tick()

    assert len(calls) == 1


def test_a_reload_recomputes_the_marks(wired):
    """Cached, but not stale: new bars must produce new marks."""
    _, live, _ = wired
    live.tick()
    before = len(live._marks["AAA"])

    live._frames = {"AAA": _frame([100.0, 105.0, 110.0, 115.0])}
    live._marks = {}                       # what a reload leaves behind
    live.tick()

    assert len(live._marks["AAA"]) > before


def test_marks_are_rebuilt_when_they_do_not_match_the_bars(wired):
    """Empty marks beside non-empty bars must not silently skip marking.

    That failure mode is dangerous precisely because it looks fine: the account
    still renders, just without any open position valued.
    """
    _, live, _ = wired
    live._marks = {}
    state = live.tick()

    assert live._marks
    assert state["equity_curve"], "no marked curve means marking was skipped"


def test_a_tick_stays_well_inside_a_one_second_budget(wired):
    """A tick slower than its interval turns the loop into a busy spin."""
    _, live, _ = wired
    live.tick()                            # warm: load frames, build the cache

    started = time.perf_counter()
    for _ in range(5):
        live.tick()
    per_tick = (time.perf_counter() - started) / 5

    assert per_tick < 0.25, f"{per_tick * 1000:.0f}ms per tick"


# ------------------------------------------------------- quote scope --------


def test_quotes_are_requested_only_for_held_symbols(wired):
    """The defect that got a live run rate-limited into no data at all.

    Marking needs a price per OPEN POSITION. Asking for the whole configured
    universe instead meant 201 symbols requested every tick to value nine.
    """
    _, live, _ = wired
    records = [
        _record(symbol="HELD_A"),
        _record(symbol="HELD_B", date="2024-01-03"),
        dict(_record(symbol="CLOSED"), outcome=1, exit_date="2024-01-05"),
    ]

    assert live._quote_symbols(records) == ["HELD_A", "HELD_B"]


def test_an_account_with_nothing_open_asks_for_no_quotes(wired):
    _, live, _ = wired
    closed = [dict(_record(), outcome=1, exit_date="2024-01-05")]

    assert live._quote_symbols(closed) == []


def test_legacy_unsized_rows_are_not_quoted(wired):
    """They are excluded from the ledger, so pricing them buys nothing."""
    _, live, _ = wired
    legacy = dict(_record(symbol="OLD"))
    legacy.pop("size_fraction")

    assert live._quote_symbols([legacy]) == []


def test_the_quote_cache_follows_the_open_book(wired):
    """A scan that opens or closes a position changes what needs pricing."""
    cfg, live, _ = wired
    live.tick()
    assert live._cache._symbols == ["AAA"]

    paper_store_path(cfg).write_text(json.dumps({"records": [
        _record(symbol="AAA"), _record(symbol="ZZZ", date="2024-01-03"),
    ]}))
    live._frames["ZZZ"] = _frame([50.0, 55.0, 60.0])
    live._marks = {}
    live.tick()

    assert live._cache._symbols == ["AAA", "ZZZ"]


def test_the_universe_is_not_what_gets_quoted(wired):
    """201 configured instruments, one open position -> one symbol quoted."""
    from titan.core.types import AssetClass, Instrument

    cfg, live, _ = wired
    cfg.universe.instruments = [
        Instrument(symbol=f"SYM{i:03d}", asset_class=AssetClass.EQUITY)
        for i in range(201)
    ]
    live.tick()

    assert live._cache._symbols == ["AAA"]
    assert live.status()["quotes"]["n_symbols"] == 1
