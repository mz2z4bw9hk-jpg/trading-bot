"""The forward paper account: signals replayed into dollars.

``titan track`` grades predictions; this turns them into a balance. The
arithmetic has to be right in the boring ways — compounding, costs, sizing off
the equity standing at entry — because an equity curve is the number people
act on and it is the easiest one to quietly get wrong.
"""

from __future__ import annotations

import math

import pytest

from titan.monitor.account import DEFAULT_STARTING_EQUITY, replay


def _record(
    symbol="AAA", date="2024-01-02", exit_date="2024-01-05",
    ret=0.10, size=0.10, cost=0.0, outcome=1, touch="tp", entry_price=100.0,
):
    return {
        "symbol": symbol, "date": date, "exit_date": exit_date,
        "outcome": outcome, "touch": touch, "ret": ret, "bars_held": 3,
        "side": "long", "size_fraction": size, "cost_estimate": cost,
        "entry_price": entry_price, "exit_price": entry_price * math.exp(ret),
        "stop_loss": entry_price * 0.98, "take_profit_levels": [entry_price * 1.1],
    }


# ---------------------------------------------------------------- basics --


def test_an_empty_log_is_a_full_untouched_account():
    state = replay([])
    assert state["equity"] == DEFAULT_STARTING_EQUITY
    assert state["starting_equity"] == 1_000_000.0
    assert state["n_closed"] == 0 and state["n_open"] == 0
    assert state["total_return"] == 0.0
    assert state["win_rate"] is None  # no trades: not "0%"


def test_a_winning_trade_moves_equity_by_notional_times_return():
    # 10% of 1,000,000 committed; a +0.10 LOG return is +10.517% simple.
    state = replay([_record(ret=0.10, size=0.10)])

    expected_pnl = 1_000_000 * 0.10 * math.expm1(0.10)
    # Dollar figures are rounded to cents on the way out.
    assert state["realized_pnl"] == pytest.approx(expected_pnl, abs=0.01)
    assert state["equity"] == pytest.approx(1_000_000 + expected_pnl, abs=0.01)
    assert state["n_closed"] == 1 and state["win_rate"] == 1.0


def test_log_returns_are_converted_before_compounding():
    """Treating the labeller's log return as simple would understate wins."""
    state = replay([_record(ret=0.20, size=1.0, cost=0.0)])
    naive = 1_000_000 * 0.20
    assert state["realized_pnl"] > naive


def test_costs_come_out_of_the_return():
    gross = replay([_record(ret=0.05, size=0.5, cost=0.0)])["realized_pnl"]
    net = replay([_record(ret=0.05, size=0.5, cost=0.004)])["realized_pnl"]
    assert net == pytest.approx(gross - 1_000_000 * 0.5 * 0.004, abs=0.01)


def test_a_loss_reduces_equity_and_is_counted_as_a_loss():
    state = replay([_record(ret=-0.05, outcome=0, touch="sl")])
    assert state["realized_pnl"] < 0
    assert state["equity"] < 1_000_000
    assert state["win_rate"] == 0.0
    assert state["trades"][0]["exit_reason"] == "sl"


# ------------------------------------------------------------ compounding --


def test_the_second_trade_is_sized_off_the_equity_the_first_produced():
    """Sizing off the starting balance forever would understate compounding."""
    first = _record(symbol="AAA", date="2024-01-02", exit_date="2024-01-03", ret=0.10)
    second = _record(symbol="BBB", date="2024-01-04", exit_date="2024-01-05", ret=0.10)

    state = replay([first, second])
    equity_after_first = state["trades"][-1]["equity_after"]  # trades are newest-first

    notional_2 = state["trades"][0]["notional"]
    assert notional_2 == pytest.approx(equity_after_first * 0.10, abs=0.01)
    assert notional_2 > 100_000  # strictly more than 10% of the starting balance


def test_equity_curve_follows_the_closes_in_order():
    recs = [
        _record(symbol="A", date="2024-01-02", exit_date="2024-01-04", ret=0.05),
        _record(symbol="B", date="2024-01-03", exit_date="2024-01-06", ret=-0.05,
                outcome=0, touch="sl"),
    ]
    curve = replay(recs)["equity_curve"]

    assert [p["date"] for p in curve] == ["2024-01-04", "2024-01-06"]
    # The win lifts equity, the loss takes some back. Both trades are ~5% of a
    # 10% position, so the pair nets out slightly ABOVE the start — the point
    # is the ordering and the direction of each step, not the sign of the sum.
    assert curve[0]["equity"] > 1_000_000
    assert curve[1]["equity"] < curve[0]["equity"]


# ----------------------------------------------------------- open + skips --


def test_an_unresolved_prediction_is_an_open_position_not_a_trade():
    rec = _record()
    rec.update(outcome=None, touch=None, ret=None, exit_date=None)

    state = replay([rec])

    assert state["n_open"] == 1 and state["n_closed"] == 0
    assert state["equity"] == 1_000_000          # nothing realized yet
    assert state["open_positions"][0]["symbol"] == "AAA"
    assert state["open_notional"] == pytest.approx(100_000.0)


def test_positions_beyond_the_gross_exposure_cap_are_recorded_not_taken():
    """An account that cannot fund a position does not silently lever up."""
    recs = [
        _record(symbol=f"S{i}", date=f"2024-01-0{i+1}", exit_date=None, size=0.30)
        for i in range(5)
    ]
    for r in recs:
        r.update(outcome=None, ret=None, exit_date=None)

    state = replay(recs, max_gross_exposure=1.0)

    assert state["n_open"] == 3                       # 0.30 x 3 = 0.90 fits, 4th does not
    assert state["n_skipped_exposure"] == 2
    assert state["skipped"][0]["reason"].startswith("would exceed")


def test_a_skipped_open_does_not_later_close():
    """Its resolution must not book P&L for a position never taken."""
    taken = _record(symbol="A", date="2024-01-02", exit_date="2024-01-09", size=0.9)
    skipped = _record(symbol="B", date="2024-01-03", exit_date="2024-01-04",
                      size=0.9, ret=0.5)
    taken.update(outcome=None, ret=None, exit_date=None)

    state = replay([taken, skipped], max_gross_exposure=1.0)

    assert state["n_skipped_exposure"] == 1
    assert state["n_closed"] == 0
    assert state["equity"] == 1_000_000


# --------------------------------------------------------------- legacy ----


def test_predictions_logged_before_account_tracking_are_excluded_not_guessed():
    legacy = {"symbol": "OLD", "date": "2023-01-02", "outcome": 1, "ret": 0.2,
              "touch": "tp", "bars_held": 3}
    state = replay([legacy, _record()])

    assert state["n_legacy_unsized"] == 1
    assert state["n_closed"] == 1                # only the sized one


def test_starting_equity_is_configurable():
    state = replay([], starting_equity=250_000.0)
    assert state["equity"] == 250_000.0


# ------------------------------------------------------- summary numbers ---


def test_profit_factor_and_averages():
    recs = [
        _record(symbol="W1", date="2024-01-02", exit_date="2024-01-03", ret=0.10),
        _record(symbol="W2", date="2024-01-04", exit_date="2024-01-05", ret=0.10),
        _record(symbol="L1", date="2024-01-06", exit_date="2024-01-07", ret=-0.05,
                outcome=0, touch="sl"),
    ]
    state = replay(recs)

    assert state["win_rate"] == pytest.approx(2 / 3, abs=1e-4)  # rounded to 4dp
    assert state["profit_factor"] > 1
    assert state["avg_win"] > 0 > state["avg_loss"]


def test_max_drawdown_is_measured_from_the_running_peak():
    recs = [
        _record(symbol="A", date="2024-01-02", exit_date="2024-01-03", ret=0.20, size=1.0),
        _record(symbol="B", date="2024-01-04", exit_date="2024-01-05", ret=-0.10,
                size=1.0, outcome=0, touch="sl"),
    ]
    state = replay(recs)
    assert state["max_drawdown"] < 0
    assert state["max_drawdown"] == pytest.approx(math.expm1(-0.10), abs=1e-5)


# ------------------------------------------------------------- leverage --


def _levered(leverage=3.0, entry_price=100.0, mmr=0.005, **kwargs):
    """A record carrying the margin terms the scanner would have written."""
    rec = _record(entry_price=entry_price, **kwargs)
    rec["leverage"] = leverage
    rec["margin_fraction"] = rec["size_fraction"] / leverage
    rec["liquidation_price"] = entry_price * (1 - (1 / leverage - mmr))
    rec["asset_class"] = "crypto"
    return rec


def test_a_levered_position_ties_up_only_its_margin():
    """3x notional posts a third of the cash. That is the entire point of it."""
    state = replay([_levered(leverage=3.0, size=0.30, outcome=None, exit_date=None)])

    assert state["n_open"] == 1
    assert state["open_notional"] == pytest.approx(300_000.0)
    assert state["open_margin"] == pytest.approx(100_000.0)
    assert state["account_leverage"] == pytest.approx(0.30)


def test_leverage_multiplies_the_pnl_on_the_same_move():
    spot = replay([_record(ret=0.05, size=0.10)])
    lev = replay([_levered(leverage=3.0, ret=0.05, size=0.30)])

    assert lev["realized_pnl"] == pytest.approx(3 * spot["realized_pnl"], rel=1e-6)


def test_a_position_cannot_lose_more_than_the_margin_behind_it():
    """The failure this guard exists for.

    A 3x position down 50% is arithmetically -150% of notional. No account
    produces that number: the exchange closes the trade at the liquidation
    price and takes the margin. Without the cap the ledger would post a loss
    larger than the cash the position ever had.
    """
    state = replay([_levered(leverage=3.0, ret=-0.70, size=0.30, outcome=0, touch="sl")])

    margin = 1_000_000 * 0.10
    assert state["realized_pnl"] == pytest.approx(-margin, abs=0.01)
    assert state["n_liquidated"] == 1
    assert state["trades"][0]["exit_reason"] == "liquidated"
    assert state["trades"][0]["liquidated"] is True


def test_a_loss_short_of_liquidation_is_posted_in_full():
    state = replay([_levered(leverage=3.0, ret=-0.05, size=0.30, outcome=0, touch="sl")])

    expected = 1_000_000 * 0.30 * math.expm1(-0.05)
    assert state["realized_pnl"] == pytest.approx(expected, abs=0.01)
    assert state["n_liquidated"] == 0
    assert state["trades"][0]["liquidated"] is False


def test_spot_positions_are_never_liquidated():
    """A cash position down 70% is down 70%, not wiped."""
    state = replay([_record(ret=-1.20, size=0.30, outcome=0, touch="sl")])
    assert state["n_liquidated"] == 0
    assert state["realized_pnl"] < 0


def test_return_on_margin_is_reported_alongside_return_on_notional():
    state = replay([_levered(leverage=3.0, ret=0.05, size=0.30)])
    row = state["trades"][0]

    assert row["return_on_margin"] == pytest.approx(3 * row["net_return"], rel=1e-3)
    assert row["leverage"] == 3.0
    assert row["margin"] == pytest.approx(100_000.0)


def test_the_account_leverage_cap_refuses_the_position_that_breaches_it():
    """Per-position limits cannot see the book. This is what does."""
    records = [
        _levered(symbol=f"C{i}", leverage=3.0, size=0.60,
                 date=f"2024-01-0{i + 1}", outcome=None, exit_date=None)
        for i in range(4)
    ]
    state = replay(records, max_gross_exposure=1.0, max_account_leverage=2.0)

    assert state["n_open"] == 3               # 3 x 60% notional = 180% <= 200%
    assert state["n_skipped_exposure"] == 1
    assert "account leverage" in state["skipped"][0]["reason"]


def test_margin_not_notional_is_what_the_cash_cap_measures():
    """Six 3x positions at 30% notional each: 180% exposure on 60% of cash.

    Charging the cash cap on notional would refuse four of these, which would
    be the account declining trades it can perfectly well fund.
    """
    records = [
        _levered(symbol=f"C{i}", leverage=3.0, size=0.30,
                 date=f"2024-01-0{i + 1}", outcome=None, exit_date=None)
        for i in range(6)
    ]
    state = replay(records, max_gross_exposure=1.0, max_account_leverage=10.0)

    assert state["n_open"] == 6
    assert state["open_margin"] == pytest.approx(600_000.0)
    assert state["open_notional"] == pytest.approx(1_800_000.0)


def test_pnl_is_split_by_asset_class():
    state = replay([
        _record(symbol="AAPL", ret=0.05, size=0.10),
        _levered(symbol="BTC-USD", leverage=3.0, ret=0.05, size=0.30),
    ])
    by_class = state["by_asset_class"]

    assert set(by_class) == {"equity", "crypto"}
    assert by_class["crypto"]["pnl"] == pytest.approx(3 * by_class["equity"]["pnl"], rel=1e-4)


def test_records_without_leverage_fields_replay_as_cash():
    """Logs written before margin existed must not be reinterpreted."""
    state = replay([_record(ret=0.05, size=0.10)])
    row = state["trades"][0]

    assert row["leverage"] == 1.0
    assert row["margin"] == row["notional"]
    assert state["n_liquidated"] == 0
