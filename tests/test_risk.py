"""Risk engine: sizing math and portfolio-level vetoes."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from titan.backtest.engine import PortfolioSnapshot, TradePlan
from titan.core.config import RiskConfig
from titan.core.types import AssetClass, Instrument, Universe
from titan.risk.portfolio import RiskEngine
from titan.risk.sizing import atr_risk_size, drawdown_throttle, fractional_kelly, vol_target_size


def test_kelly_formula_and_cap():
    # p=0.6, b=2 -> f* = (0.6*2-0.4)/2 = 0.4; quarter-kelly = 0.1
    assert fractional_kelly(0.6, 2.0, 0.25, cap=1.0) == pytest.approx(0.1)
    assert fractional_kelly(0.6, 2.0, 0.25, cap=0.05) == 0.05
    assert fractional_kelly(0.4, 1.0, 0.25, cap=1.0) == 0.0  # negative edge -> 0
    assert fractional_kelly(0.0, 2.0, 0.25, cap=1.0) == 0.0


def test_vol_target_size():
    # budget = 0.12/sqrt(4) = 0.06; asset vol 0.30 -> weight 0.2
    assert vol_target_size(0.30, 0.12, 4, cap=1.0) == pytest.approx(0.2)
    assert vol_target_size(0.0, 0.12, 4, cap=1.0) == 0.0


def test_atr_risk_size():
    # risk 0.5% of equity with a 2% stop -> 25% position
    assert atr_risk_size(0.02, 0.5, cap=1.0) == pytest.approx(0.25)


def test_drawdown_throttle_monotone():
    vals = [drawdown_throttle(-d, 0.05, 0.15) for d in (0.0, 0.05, 0.10, 0.15, 0.30)]
    assert vals[0] == vals[1] == 1.0
    assert vals[2] == pytest.approx(0.5)
    assert vals[3] == vals[4] == 0.0
    assert all(a >= b for a, b in itertools.pairwise(vals))


def _snapshot(**kw) -> PortfolioSnapshot:
    base = {"equity": 1e6, "n_positions": 0, "gross_exposure": 0.0, "open_risk_fraction": 0.0,
                "symbol_weights": {}, "sector_weights": {}, "strategy_drawdown": 0.0}
    base.update(kw)
    return PortfolioSnapshot(**base)


def _plan(symbol="AAA", size=0.10) -> TradePlan:
    return TradePlan(symbol=symbol, decision_date=pd.Timestamp("2022-06-01", tz="UTC"),
                     size_fraction=size, stop_price=95.0, tp_price=110.0,
                     max_holding_bars=10, entry_ref=100.0)


def _universe() -> Universe:
    return Universe(instruments=[
        Instrument("AAA", AssetClass.EQUITY, "tech"),
        Instrument("BBB", AssetClass.EQUITY, "tech"),
    ])


def test_crash_regime_zeroes_size():
    regimes = pd.Series(["crash"], index=[pd.Timestamp("2022-05-31", tz="UTC")])
    eng = RiskEngine(RiskConfig(), regimes=regimes)
    assert eng.approve(_plan(), _snapshot()) == 0.0


def test_bear_regime_scales_down():
    regimes = pd.Series(["bear"], index=[pd.Timestamp("2022-05-31", tz="UTC")])
    eng = RiskEngine(RiskConfig(), regimes=regimes)
    approved = eng.approve(_plan(size=0.10), _snapshot())
    assert approved == pytest.approx(0.10 * 0.30, rel=1e-6)


def test_heat_cap_binds():
    cfg = RiskConfig(portfolio_heat_cap_pct=1.0)  # 1% total open risk
    eng = RiskEngine(cfg)
    # candidate risk = size * stop_dist = 0.10 * 5% = 0.5%; existing 0.8% -> only 0.2% room
    snap = _snapshot(open_risk_fraction=0.008)
    approved = eng.approve(_plan(size=0.10), snap)
    assert approved == pytest.approx(0.10 * (0.002 / 0.005), rel=1e-6)
    # no room at all
    snap_full = _snapshot(open_risk_fraction=0.011)
    assert eng.approve(_plan(size=0.10), snap_full) == 0.0


def test_sector_cap_binds():
    cfg = RiskConfig(max_sector_weight=0.20)
    eng = RiskEngine(cfg, universe=_universe())
    snap = _snapshot(sector_weights={"tech": 0.15}, symbol_weights={"BBB": 0.15})
    assert eng.approve(_plan("AAA", size=0.10), snap) == pytest.approx(0.05, abs=1e-9)
    snap_full = _snapshot(sector_weights={"tech": 0.25})
    assert eng.approve(_plan("AAA", size=0.10), snap_full) == 0.0


def test_correlation_penalty():
    idx = pd.bdate_range("2022-01-03", periods=120, tz="UTC")
    rng = np.random.default_rng(0)
    base = rng.normal(0, 0.01, len(idx))
    returns = pd.DataFrame({
        "AAA": base + rng.normal(0, 0.001, len(idx)),   # ~0.99 corr with BBB
        "BBB": base + rng.normal(0, 0.001, len(idx)),
        "ZZZ": rng.normal(0, 0.01, len(idx)),
    }, index=idx)
    cfg = RiskConfig(correlation_penalty_threshold=0.6)
    eng = RiskEngine(cfg, returns=returns)
    snap = _snapshot(symbol_weights={"BBB": 0.1})
    correlated = eng.approve(_plan("AAA", size=0.10), snap)
    uncorrelated = eng.approve(_plan("ZZZ", size=0.10), snap)
    assert correlated < uncorrelated
    assert correlated == pytest.approx(0.05, abs=0.01)  # ~50% haircut at rho~1


def test_dd_throttle_in_approve():
    eng = RiskEngine(RiskConfig(dd_throttle_start=0.05, dd_throttle_full=0.15))
    healthy = eng.approve(_plan(size=0.10), _snapshot(strategy_drawdown=0.0))
    hurting = eng.approve(_plan(size=0.10), _snapshot(strategy_drawdown=-0.10))
    dead = eng.approve(_plan(size=0.10), _snapshot(strategy_drawdown=-0.20))
    assert healthy == pytest.approx(0.10)
    assert hurting == pytest.approx(0.05)
    assert dead == 0.0


def test_portfolio_var_cvar():
    idx = pd.bdate_range("2021-01-01", periods=300, tz="UTC")
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({"AAA": rng.normal(0, 0.02, 300)}, index=idx)
    eng = RiskEngine(RiskConfig(), returns=returns)
    var, cvar = eng.portfolio_var_cvar({"AAA": 0.5}, idx[-1])
    assert var < 0 and cvar <= var
    assert abs(var) < 0.05  # half a position of 2% daily vol
