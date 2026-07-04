"""Monte Carlo machinery: statistical sanity of the resamplers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.backtest.monte_carlo import (
    bootstrap_analysis,
    risk_of_ruin,
    stationary_block_bootstrap,
)


def test_block_bootstrap_shape_and_values():
    values = np.arange(100, dtype=float)
    sims = stationary_block_bootstrap(values, n_sims=50, avg_block=8, seed=1)
    assert sims.shape == (50, 100)
    assert set(np.unique(sims)).issubset(set(values))


def test_block_bootstrap_preserves_mean():
    rng = np.random.default_rng(2)
    values = rng.normal(0.001, 0.01, 2000)
    sims = stationary_block_bootstrap(values, n_sims=300, avg_block=10, seed=3)
    assert sims.mean() == pytest.approx(values.mean(), abs=3e-4)


def test_block_bootstrap_preserves_autocorrelation():
    """Blocks must keep short-range serial structure that iid resampling destroys."""
    rng = np.random.default_rng(4)
    n = 3000
    x = np.zeros(n)
    eps = rng.standard_normal(n)
    for t in range(1, n):
        x[t] = 0.6 * x[t - 1] + eps[t]

    def lag1(m: np.ndarray) -> float:
        a, b = m[:, :-1].ravel(), m[:, 1:].ravel()
        return float(np.corrcoef(a, b)[0, 1])

    block = stationary_block_bootstrap(x, n_sims=20, avg_block=25, seed=5)
    iid = stationary_block_bootstrap(x, n_sims=20, avg_block=1.0000001, seed=5)
    assert lag1(block) > 0.4
    assert abs(lag1(iid)) < 0.15


def test_bootstrap_analysis_ci_contains_point_estimate():
    rng = np.random.default_rng(6)
    returns = pd.Series(rng.normal(0.0006, 0.008, 1200))
    rep = bootstrap_analysis(returns, n_sims=400, seed=7)
    point_sharpe = float(returns.mean() / returns.std() * np.sqrt(252))
    lo, hi = rep.sharpe_ci
    assert lo < point_sharpe < hi
    assert 0 <= rep.p_sharpe_positive <= 1
    assert rep.max_dd_ci[0] <= rep.max_dd_ci[1] <= 0


def test_risk_of_ruin_monotone_in_edge():
    rng = np.random.default_rng(8)
    good = rng.normal(0.004, 0.02, 300)
    bad = rng.normal(-0.004, 0.02, 300)
    r_good = risk_of_ruin(good, trades_per_year=100, seed=9)["risk_of_ruin"]
    r_bad = risk_of_ruin(bad, trades_per_year=100, seed=9)["risk_of_ruin"]
    assert r_bad > r_good


def test_risk_of_ruin_insufficient_trades():
    out = risk_of_ruin(np.array([0.01, -0.01]), trades_per_year=10)
    assert np.isnan(out["risk_of_ruin"])
