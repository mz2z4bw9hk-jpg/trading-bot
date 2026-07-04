"""Backtest engine: cash accounting proven by hand, pessimism verified."""

from __future__ import annotations

import pandas as pd
import pytest

from titan.backtest.costs import CostModel
from titan.backtest.engine import BacktestEngine, TradePlan
from titan.core.config import BacktestConfig, CostConfig


def _frame(prices: list[tuple[float, float, float, float]]) -> pd.DataFrame:
    idx = pd.bdate_range("2022-01-03", periods=len(prices), tz="UTC")
    df = pd.DataFrame(prices, columns=["open", "high", "low", "close"], index=idx)
    df["volume"] = 1e6
    return df


def _cfg(**kw) -> BacktestConfig:
    costs = CostConfig(commission_bps=10.0, spread_bps=0.0, impact_coefficient=0.0)
    return BacktestConfig(initial_capital=100_000.0, costs=costs, **kw)


def test_single_trade_accounting_by_hand():
    # Decision bar 0, entry at bar 1 open=100, TP=110 touched bar 3, commission 10bps each way.
    frame = _frame([
        (100, 101, 99, 100),
        (100, 104, 99, 103),
        (103, 108, 102, 107),
        (107, 111, 106, 108),   # high 111 >= tp 110
        (108, 109, 107, 108),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.5,
                     stop_price=90.0, tp_price=110.0, max_holding_bars=10, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])

    assert len(result.trades) == 1
    t = result.trades[0]
    fill_in = 100 * 1.001            # +10bps
    shares = 0.5 * 100_000 / fill_in
    fill_out = 110 * 0.999           # -10bps
    expected_pnl = shares * (fill_out - fill_in)
    assert t.exit_reason == "tp"
    assert t.entry_price == pytest.approx(fill_in)
    assert t.exit_price == pytest.approx(fill_out)
    assert t.pnl_cash == pytest.approx(expected_pnl, rel=1e-9)
    assert result.equity.iloc[-1] == pytest.approx(100_000 + expected_pnl, rel=1e-9)


def test_stop_wins_on_ambiguous_bar():
    # bar 2 touches BOTH stop (95) and tp (105): pessimistic rule -> stop.
    frame = _frame([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 106, 94, 100),
        (100, 101, 99, 100),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=95.0, tp_price=105.0, max_holding_bars=10, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].exit_price == pytest.approx(95.0 * 0.999)


def test_gap_through_stop_fills_at_open_not_stop():
    frame = _frame([
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (90, 92, 88, 91),      # gaps below the 95 stop -> fill at open 90
        (91, 92, 90, 91),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=95.0, tp_price=120.0, max_holding_bars=10, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])
    assert result.trades[0].exit_reason == "stop"
    assert result.trades[0].exit_price == pytest.approx(90.0 * 0.999)


def test_time_exit_and_mae_mfe():
    frame = _frame([
        (100, 101, 99, 100),
        (100, 102, 98, 101),
        (101, 103, 97, 102),
        (102, 104, 100, 103),
        (103, 105, 101, 104),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=80.0, tp_price=150.0, max_holding_bars=3, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])
    t = result.trades[0]
    assert t.exit_reason == "time"
    assert t.bars_held == 3
    # excursions span entry bar through exit bar: lows 98,97,100,101 / highs 102,103,104,105
    assert t.mae == pytest.approx(97 / t.entry_price - 1)
    assert t.mfe == pytest.approx(105 / t.entry_price - 1)


def test_max_positions_enforced():
    frames = {}
    plans = []
    base = _frame([(100, 101, 99, 100)] * 6)
    for i in range(4):
        sym = f"S{i}"
        frames[sym] = base.copy()
        plans.append(TradePlan(symbol=sym, decision_date=base.index[0], size_fraction=0.1,
                               stop_price=90, tp_price=120, max_holding_bars=4, entry_ref=100))
    cfg = _cfg(max_positions=2)
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    result = engine.run(frames, plans)
    assert len(result.trades) == 2
    assert result.n_rejected == 2


def test_gross_exposure_cap_scales_size():
    base = _frame([(100, 101, 99, 100)] * 6)
    frames = {"A": base.copy(), "B": base.copy()}
    plans = [
        TradePlan(symbol="A", decision_date=base.index[0], size_fraction=0.8,
                  stop_price=90, tp_price=120, max_holding_bars=4, entry_ref=100),
        TradePlan(symbol="B", decision_date=base.index[0], size_fraction=0.8,
                  stop_price=90, tp_price=120, max_holding_bars=4, entry_ref=100),
    ]
    cfg = _cfg(max_gross_exposure=1.0)
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    result = engine.run(frames, plans)
    assert len(result.trades) == 2
    # second position squeezed into remaining ~20% exposure
    sizes = sorted(t.size_fraction for t in result.trades)
    assert sizes[0] == pytest.approx(0.2, abs=0.02)
    assert sizes[1] == pytest.approx(0.8, abs=0.01)


def test_execution_is_next_bar_never_same_bar():
    frame = _frame([(100, 150, 99, 149), (110, 111, 109, 110), (110, 111, 109, 110)])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=90, tp_price=140, max_holding_bars=2, entry_ref=100)
    result = engine.run({"X": frame}, [plan])
    t = result.trades[0]
    # bar-0 high of 150 (> tp) must NOT fill: entry only happens at bar 1 open=110
    assert t.entry_date == frame.index[1]
    assert t.entry_price == pytest.approx(110 * 1.001)


def test_stop_live_on_entry_bar():
    """Labels count barrier touches from the entry bar; the engine must too."""
    frame = _frame([
        (100, 101, 99, 100),
        (100, 101, 94, 96),    # entry at open 100; low 94 pierces the 95 stop same bar
        (96, 97, 95, 96),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=95.0, tp_price=120.0, max_holding_bars=5, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])
    t = result.trades[0]
    assert t.exit_reason == "stop"
    assert t.entry_date == t.exit_date == frame.index[1]
    assert t.bars_held == 0
    assert t.exit_price == pytest.approx(95.0 * 0.999)


def test_gap_past_level_invalidates_entry():
    """If the open already sits beyond stop or target, the plan is stale: no fill."""
    frame = _frame([
        (100, 101, 99, 100),
        (94, 95, 93, 94),      # opens below the 95 stop -> plan invalidated
        (94, 95, 93, 94),
    ])
    cfg = _cfg()
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    plan = TradePlan(symbol="X", decision_date=frame.index[0], size_fraction=0.2,
                     stop_price=95.0, tp_price=120.0, max_holding_bars=5, entry_ref=100.0)
    result = engine.run({"X": frame}, [plan])
    assert result.trades == []
    assert result.n_rejected == 1


def test_capacity_goes_to_highest_priority():
    """With one slot and two same-day plans, the higher-priority plan must fill."""
    base = _frame([(100, 101, 99, 100)] * 5)
    frames = {"AAA": base.copy(), "ZZZ": base.copy()}
    mk = lambda sym, prio: TradePlan(  # noqa: E731
        symbol=sym, decision_date=base.index[0], size_fraction=0.1,
        stop_price=90, tp_price=120, max_holding_bars=3, entry_ref=100, priority=prio)
    cfg = _cfg(max_positions=1)
    engine = BacktestEngine(cfg, CostModel(cfg.costs))
    # AAA sorts first alphabetically but ZZZ carries higher priority
    result = engine.run(frames, [mk("AAA", 10.0), mk("ZZZ", 90.0)])
    assert len(result.trades) == 1
    assert result.trades[0].symbol == "ZZZ"
