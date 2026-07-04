"""Metrics: verified against hand computations, not against themselves."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.backtest.engine import TradeRecord
from titan.backtest.metrics import (
    cagr,
    deflated_sharpe_ratio,
    max_drawdown,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sortino_ratio,
    summarize,
    var_cvar,
)


def _trade(pnl_fraction: float) -> TradeRecord:
    ts = pd.Timestamp("2021-01-04", tz="UTC")
    return TradeRecord(
        symbol="X", entry_date=ts, exit_date=ts, entry_price=100, exit_price=100 * (1 + pnl_fraction),
        shares=1, size_fraction=0.1, pnl_cash=pnl_fraction * 100, pnl_fraction=pnl_fraction,
        bars_held=3, exit_reason="tp", mae=-0.01, mfe=0.03,
    )


def test_sharpe_hand_computed():
    r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.007])
    expected = r.mean() / r.std() * np.sqrt(252)
    assert sharpe_ratio(r) == pytest.approx(expected)


def test_sortino_uses_downside_only():
    r = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    downside = np.sqrt(np.mean([0.02**2, 0.01**2]))
    assert sortino_ratio(r) == pytest.approx(r.mean() / downside * np.sqrt(252))


def test_max_drawdown_hand_computed():
    equity = pd.Series([100, 120, 90, 95, 130, 110])
    dd, longest = max_drawdown(equity)
    assert dd == pytest.approx(90 / 120 - 1)      # -25%
    assert longest == 2                            # bars 90,95 below the 120 peak


def test_cagr_two_years():
    equity = pd.Series(np.linspace(0, 1, 505) * 0 + 1.0)
    equity.iloc[-1] = 1.21
    assert cagr(equity) == pytest.approx(0.1, abs=1e-3)  # sqrt(1.21)-1


def test_var_cvar_tail():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 5000))
    var, cvar = var_cvar(r, 0.95)
    assert var == pytest.approx(np.quantile(r, 0.05))
    assert cvar <= var  # expected shortfall at least as bad as VaR


def test_profit_factor_expectancy_win_rate():
    trades = [_trade(0.02), _trade(0.02), _trade(-0.01)]
    equity = pd.Series([1.0, 1.01, 1.02, 1.03], index=pd.RangeIndex(4))
    s = summarize(equity, trades)
    assert s.win_rate == pytest.approx(2 / 3)
    assert s.profit_factor == pytest.approx(0.04 / 0.01)
    assert s.expectancy == pytest.approx((0.02 + 0.02 - 0.01) / 3)
    assert s.avg_win == pytest.approx(0.02)
    assert s.avg_loss == pytest.approx(-0.01)


def test_psr_increases_with_n():
    lo = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=50, skew=0, kurtosis=3)
    hi = probabilistic_sharpe_ratio(0.1, 0.0, n_obs=1000, skew=0, kurtosis=3)
    assert hi > lo > 0.5


def test_dsr_decreases_with_trials():
    kwargs = {"observed_sr": 0.08, "sr_variance_across_trials": 0.002,
              "n_obs": 500, "skew": 0.0, "kurtosis": 3.0}
    d1 = deflated_sharpe_ratio(n_trials=1, **kwargs)
    d10 = deflated_sharpe_ratio(n_trials=10, **kwargs)
    d100 = deflated_sharpe_ratio(n_trials=100, **kwargs)
    assert d1 > d10 > d100  # more trials -> more deflation


def test_negative_skew_fat_tails_reduce_psr():
    normal = probabilistic_sharpe_ratio(0.1, 0.0, 500, skew=0.0, kurtosis=3.0)
    ugly = probabilistic_sharpe_ratio(0.1, 0.0, 500, skew=-1.5, kurtosis=8.0)
    assert ugly < normal
