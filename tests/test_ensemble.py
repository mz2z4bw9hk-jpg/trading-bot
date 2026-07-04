"""Ensemble: finds planted signal, stays honest on noise, calibrates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from titan.core.config import ModelConfig
from titan.models.ensemble import CalibratedEnsemble


def _make_data(n=3000, n_feat=8, signal=1.2, seed=0):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.standard_normal((n, n_feat)), columns=[f"f{i}" for i in range(n_feat)])
    logits = signal * X["f0"] - 0.7 * signal * X["f1"] + rng.standard_normal(n)
    y = pd.Series((logits > 0).astype(int))
    dates = pd.DatetimeIndex(pd.bdate_range("2015-01-01", periods=n, tz="UTC"))
    return X, y, dates


@pytest.fixture(scope="module")
def fitted():
    X, y, dates = _make_data()
    cut = 2400
    model = CalibratedEnsemble(
        ModelConfig(members=["hgb", "logistic"], tuning_iterations=0), horizon_bars=5, seed=1
    )
    model.fit(X.iloc[:cut], y.iloc[:cut], dates[:cut])
    return model, X.iloc[cut:], y.iloc[cut:]


def test_finds_planted_signal(fitted):
    model, X_te, y_te = fitted
    p = model.predict_proba(X_te)[:, 1]
    assert roc_auc_score(y_te, p) > 0.75


def test_probabilities_are_calibrated(fitted):
    model, X_te, y_te = fitted
    p = model.predict_proba(X_te)[:, 1]
    brier = np.mean((p - y_te.to_numpy()) ** 2)
    assert brier < 0.22  # meaningfully better than the 0.25 coin-flip bound
    # bucket check: high-p bucket must hit more often than low-p bucket
    hi, lo = p >= np.quantile(p, 0.8), p <= np.quantile(p, 0.2)
    assert y_te[hi].mean() > y_te[lo].mean() + 0.2


def test_no_skill_claimed_on_shuffled_labels():
    X, y, dates = _make_data(seed=2)
    rng = np.random.default_rng(3)
    y_shuffled = pd.Series(rng.permutation(y.to_numpy()))
    cut = 2400
    model = CalibratedEnsemble(
        ModelConfig(members=["hgb", "logistic"], tuning_iterations=0), horizon_bars=5, seed=1
    )
    model.fit(X.iloc[:cut], y_shuffled.iloc[:cut], dates[:cut])
    p = model.predict_proba(X.iloc[cut:])[:, 1]
    auc = roc_auc_score(y_shuffled.iloc[cut:], p)
    assert 0.42 < auc < 0.58  # no fabricated edge on pure noise


def test_uncertainty_higher_off_manifold(fitted):
    model, X_te, _ = fitted
    unc_in = model.uncertainty(X_te).mean()
    X_far = X_te * 8.0  # far outside the training distribution
    unc_out = model.uncertainty(X_far).mean()
    assert unc_out > unc_in


def test_predict_proba_contract(fitted):
    model, X_te, _ = fitted
    proba = model.predict_proba(X_te)
    assert proba.shape == (len(X_te), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert ((proba > 0) & (proba < 1)).all()
    # column order must match classes_
    assert list(model.classes_) == [0, 1]


def test_column_order_robustness(fitted):
    model, X_te, _ = fitted
    shuffled_cols = X_te[list(reversed(X_te.columns))]
    p1 = model.predict_proba(X_te)[:, 1]
    p2 = model.predict_proba(shuffled_cols)[:, 1]
    assert np.allclose(p1, p2)  # reindex must restore training order


def test_member_weights_favor_better_member(fitted):
    model, _, _ = fitted
    report = model.report_
    assert report is not None
    weights = {m.name: m.weight for m in report.members}
    losses = {m.name: m.calib_log_loss for m in report.members}
    best = min(losses, key=losses.get)
    assert weights[best] == max(weights.values())
    assert sum(weights.values()) == pytest.approx(1.0)


def test_tuning_path_samples_all_members():
    """Regression: hyperparameter sampling must produce valid params for every
    member (numpy's choice() once coerced rf max_features to a string)."""
    X, y, dates = _make_data(n=900, seed=5)
    model = CalibratedEnsemble(
        ModelConfig(members=["hgb", "rf", "logistic"], tuning_iterations=2),
        horizon_bars=5,
        seed=2,
    )
    model.fit(X, y, dates)
    p = model.predict_proba(X.tail(100))[:, 1]
    assert np.isfinite(p).all()
