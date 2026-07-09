"""Ensemble: finds planted signal, stays honest on noise, calibrates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from titan.core.config import ModelConfig
from titan.models.ensemble import CalibratedEnsemble, venn_abers_interval


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


def test_internal_folds_are_purged():
    """No internal training event may live into its own OOF test block."""
    n = 1200
    dates = pd.DatetimeIndex(pd.bdate_range("2016-01-01", periods=n, tz="UTC"))
    # every event lasts 15 bars
    t1 = pd.Series(dates[np.minimum(np.arange(n) + 15, n - 1)])
    model = CalibratedEnsemble(
        ModelConfig(members=["logistic"], tuning_iterations=0, internal_folds=3),
        horizon_bars=15,
    )
    folds = model._internal_folds(dates, t1)
    assert len(folds) >= 2
    for train_idx, test_idx in folds:
        test_start = dates[test_idx].min()
        assert (pd.DatetimeIndex(t1.iloc[train_idx]) < test_start).all()
        assert (dates[train_idx] < test_start).all()


def test_internal_folds_shrink_on_short_windows():
    """K collapses toward 2 rather than producing sliver folds."""
    n = 400  # 400 // (3+1) = 100 dates/segment -> fine; 400 // (5+1) = 66 also fine
    dates = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=n, tz="UTC"))
    tiny = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=150, tz="UTC"))
    model = CalibratedEnsemble(
        ModelConfig(members=["logistic"], tuning_iterations=0, internal_folds=4),
        horizon_bars=5,
    )
    assert len(model._internal_folds(dates, None)) >= 2
    # 150 dates cannot host 4 folds of >=60 dates: K must shrink (and may
    # still fail the row-count floor, which raises loudly).
    try:
        folds = model._internal_folds(tiny, None)
        assert len(folds) <= 2
    except ValueError:
        pass  # acceptable: too small is a loud error, never a silent sliver


def test_oof_report_semantics(fitted):
    """n_fit is the FULL window; n_calib counts pooled OOF rows only."""
    model, _, _ = fitted
    r = model.report_
    assert r is not None
    assert r.n_fit == 2400              # entire training window
    assert 0 < r.n_calib < r.n_fit      # OOF pool is a strict subset
    assert np.isfinite(r.calib_brier)


# ---------------------------------------------------------------------- #
# Venn-ABERS intervals


def test_venn_abers_point_properties():
    rng = np.random.default_rng(11)
    scores = rng.uniform(0, 1, 800)
    labels = (rng.uniform(0, 1, 800) < scores).astype(int)  # perfectly calibrated world
    prev = (0.0, 0.0)
    for s in (0.1, 0.3, 0.5, 0.7, 0.9):
        p0, p1 = venn_abers_interval(scores, labels, s)
        assert 0.0 <= p0 <= p1 <= 1.0
        # roughly recovers the true probability and is monotone in the score
        assert p0 - 0.12 <= s <= p1 + 0.12
        assert p0 >= prev[0] - 1e-9 and p1 >= prev[1] - 1e-9
        prev = (p0, p1)


def test_venn_abers_band_tightens_with_evidence():
    """More calibration data at a score = narrower band there."""
    rng = np.random.default_rng(7)

    def width(n: int) -> float:
        scores = rng.uniform(0, 1, n)
        labels = (rng.uniform(0, 1, n) < scores).astype(int)
        p0, p1 = venn_abers_interval(scores, labels, 0.6)
        return p1 - p0

    assert width(2000) < width(60)


def test_ensemble_interval_brackets_prediction(fitted):
    model, X_te, _ = fitted
    sample = X_te.head(40)
    bands = model.probability_interval(sample)
    assert bands.shape == (40, 2)
    assert (bands[:, 0] <= bands[:, 1] + 1e-12).all()
    assert ((bands >= 0) & (bands <= 1)).all()
    # a fitted model on 2400 rows should not produce degenerate full-width bands
    assert float(np.median(bands[:, 1] - bands[:, 0])) < 0.25
    # CONTAINMENT: on the same calibration evidence, adding (s, 1) can only
    # pull the isotonic fit at s up and (s, 0) only down, so the band must
    # bracket the deployed calibrated probability (tolerance = clip bounds).
    assert model.report_ is not None and model.report_.calibration == "isotonic"
    p_hat = model.predict_proba(sample)[:, 1]
    assert (bands[:, 0] <= p_hat + 2e-3).all()
    assert (p_hat <= bands[:, 1] + 2e-3).all()


def test_interval_requires_fit():
    model = CalibratedEnsemble(
        ModelConfig(members=["logistic"], tuning_iterations=0), horizon_bars=5
    )
    assert not model.has_intervals
    with pytest.raises(RuntimeError, match="fit"):
        model.probability_interval(pd.DataFrame({"f0": [0.0]}))
