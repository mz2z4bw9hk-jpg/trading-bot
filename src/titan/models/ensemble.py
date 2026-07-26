"""Calibrated heterogeneous ensemble with honest internal validation.

Members (gradient boosting, random forest, regularized logistic regression)
are diverse by construction: different bias structures disagree in different
ways, and the disagreement itself is a usable uncertainty estimate.

Fitting protocol (all inside the *training* window of an outer fold — the
outer walk-forward never sees any of this):

1. Build K purged, forward-chaining internal folds over the training window
   (events whose life overlaps an internal test block are excluded from its
   training block, using real event end-times when provided).
2. Random-search each member's hyperparameters on the LAST internal fold —
   the one with the most history (random search is a strong baseline —
   Bergstra & Bengio 2012; the sampler is the seam where Bayesian
   optimization plugs in without touching callers).
3. Collect OUT-OF-FOLD predictions for every member across all internal
   folds. Member weights come from exponentiated negative pooled-OOF
   log-loss, and the isotonic (or Platt) calibrator is fit on the pooled
   OOF ensemble probabilities — so calibration sees several market regimes,
   not just the tail of the window. Raw scores of boosted trees are *not*
   probabilities; position sizing needs calibrated ones.
4. Refit every member on the FULL training window with its chosen
   hyperparameters. The deployed members waste no data; the calibrator and
   weights were learned strictly out-of-fold.

``predict_proba`` returns calibrated P(take-profit before stop); ``uncertainty``
returns member disagreement (std of member probabilities);
``probability_interval`` returns the inductive Venn-ABERS band [p0, p1]
(Vovk & Petej 2014) computed against the pooled OOF scores — a distribution-
free measure of how much the calibration itself can be trusted at this score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from titan.core.config import ModelConfig
from titan.core.log import get_logger
from titan.models.cv import to_naive_utc

logger = get_logger(__name__)

_MIN_CALIB_ROWS_ISOTONIC = 300
_WEIGHT_TEMPERATURE = 0.02  # log-loss units; smaller = sharper member weighting


def venn_abers_interval(
    scores: np.ndarray, labels: np.ndarray, s: float
) -> tuple[float, float]:
    """Inductive Venn-ABERS interval [p0, p1] for one test score.

    p1 refits isotonic regression on the calibration set plus (s, 1) and
    reads the fit at s; p0 does the same with (s, 0). The pair brackets the
    probability with a validity guarantee that holds regardless of the score
    distribution; the width is honest calibration uncertainty — wide where
    calibration data is thin, narrow where it is dense.
    """
    xs = np.append(scores, s)
    iso1 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso1.fit(xs, np.append(labels, 1))
    p1 = float(iso1.predict([s])[0])
    iso0 = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso0.fit(xs, np.append(labels, 0))
    p0 = float(iso0.predict([s])[0])
    return min(p0, p1), max(p0, p1)


def _sample_params(member: str, rng: np.random.Generator) -> dict[str, Any]:
    if member == "hgb":
        return {
            "max_iter": int(rng.choice([100, 200, 300])),
            "learning_rate": float(np.exp(rng.uniform(np.log(0.03), np.log(0.15)))),
            "max_leaf_nodes": int(rng.choice([15, 31, 63])),
            "min_samples_leaf": int(rng.choice([20, 50, 100])),
            "l2_regularization": float(np.exp(rng.uniform(np.log(1e-3), np.log(1.0)))),
        }
    if member == "rf":
        # NOTE: numpy's choice() on a mixed list coerces to str; index instead.
        max_features_options: list[str | float] = ["sqrt", 0.3]
        return {
            "n_estimators": int(rng.choice([200, 300])),
            "max_depth": int(rng.choice([4, 6, 8])),
            "min_samples_leaf": int(rng.choice([20, 50, 100])),
            "max_features": max_features_options[int(rng.integers(len(max_features_options)))],
        }
    if member == "logistic":
        return {"C": float(np.exp(rng.uniform(np.log(0.01), np.log(10.0))))}
    raise ValueError(f"unknown member: {member}")


def _default_params(member: str) -> dict[str, Any]:
    table: dict[str, dict[str, Any]] = {
        "hgb": {"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 31,
                "min_samples_leaf": 50, "l2_regularization": 0.1},
        "rf": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50, "max_features": "sqrt"},
        "logistic": {"C": 1.0},
    }
    return table[member]


def _build_member(member: str, params: dict[str, Any], seed: int):
    if member == "hgb":
        return HistGradientBoostingClassifier(random_state=seed, early_stopping=False, **params)
    if member == "rf":
        mf = params.get("max_features", "sqrt")
        params = {**params, "max_features": mf if isinstance(mf, str) else float(mf)}
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestClassifier(random_state=seed, n_jobs=-1, **params)),
            ]
        )
    if member == "logistic":
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, random_state=seed, **params)),
            ]
        )
    raise ValueError(f"unknown member: {member}")


def _fit_member(model, X: pd.DataFrame, y: np.ndarray, w: np.ndarray | None):
    if w is None:
        return model.fit(X, y)
    if isinstance(model, Pipeline):
        return model.fit(X, y, model__sample_weight=w)
    return model.fit(X, y, sample_weight=w)


@dataclass(slots=True)
class MemberReport:
    name: str
    params: dict[str, Any]
    calib_log_loss: float
    calib_auc: float
    weight: float


@dataclass(slots=True)
class FitReport:
    members: list[MemberReport] = field(default_factory=list)
    calibration: str = "isotonic"
    n_fit: int = 0
    n_calib: int = 0
    calib_brier: float = float("nan")

    def to_dict(self) -> dict:
        return {
            "n_fit": self.n_fit,
            "n_calib": self.n_calib,
            "calibration": self.calibration,
            "calib_brier": None if np.isnan(self.calib_brier) else round(self.calib_brier, 5),
            "members": [
                {
                    "name": m.name,
                    "weight": round(m.weight, 4),
                    "log_loss": round(m.calib_log_loss, 5),
                    "auc": round(m.calib_auc, 4),
                    "params": m.params,
                }
                for m in self.members
            ],
        }


class CalibratedEnsemble:
    """Weighted soft-vote ensemble with out-of-sample probability calibration."""

    def __init__(self, cfg: ModelConfig, horizon_bars: int, seed: int = 7) -> None:
        self._cfg = cfg
        self._horizon = horizon_bars
        self._seed = seed
        self._members: dict[str, Any] = {}
        self._weights: dict[str, float] = {}
        self._calibrator: Any = None
        self._calibration_kind: str = cfg.calibration
        self._oof_scores: np.ndarray | None = None
        self._oof_labels: np.ndarray | None = None
        self.feature_names_: list[str] = []
        self.report_: FitReport | None = None
        self.classes_ = np.array([0, 1])

    # ------------------------------------------------------------------ #

    def _internal_folds(
        self,
        dates: pd.DatetimeIndex,
        t1: pd.Series | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """K purged forward-chaining (train_idx, test_idx) splits over train.

        Unique dates are cut into K+1 contiguous segments; fold k tests on
        segment k+1 and trains on everything strictly before it, minus any
        event whose life (real ``t1`` when given, else date + horizon bars)
        reaches into the test block.
        """
        values = to_naive_utc(dates)
        unique = np.unique(values)
        k = self._cfg.internal_folds
        # Shrink K rather than fail when the window is short.
        while k > 2 and len(unique) // (k + 1) < 60:
            k -= 1
        edges = np.linspace(0, len(unique), k + 2, dtype=int)

        if t1 is not None:
            end_values = to_naive_utc(t1)
        else:
            pos = np.searchsorted(unique, values)
            end_pos = np.minimum(pos + self._horizon, len(unique) - 1)
            end_values = unique[end_pos]

        folds: list[tuple[np.ndarray, np.ndarray]] = []
        for j in range(1, k + 1):
            test_start = unique[edges[j]]
            test_end = unique[edges[j + 1] - 1]
            train_mask = (values < test_start) & (end_values < test_start)
            test_mask = (values >= test_start) & (values <= test_end)
            train_idx, test_idx = np.where(train_mask)[0], np.where(test_mask)[0]
            if len(train_idx) >= 200 and len(test_idx) >= 50:
                folds.append((train_idx, test_idx))
        if not folds:
            raise ValueError("training window too small for internal OOF folds")
        return folds

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        sample_weight: pd.Series | None = None,
        t1: pd.Series | None = None,
    ) -> CalibratedEnsemble:
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, dates must align")
        if len(X) > self._cfg.max_train_rows:
            keep = slice(-self._cfg.max_train_rows, None)
            X, y, dates = X.iloc[keep], y.iloc[keep], dates[keep]
            if sample_weight is not None:
                sample_weight = sample_weight.iloc[keep]
            if t1 is not None:
                t1 = t1.iloc[keep]

        # Drop features that are constant in THIS training window. They carry
        # no information by construction, and the gradient-boosting binner does
        # not merely ignore them — it builds bin edges from the midpoints of
        # adjacent distinct values, so a single distinct value raises deep
        # inside sklearn ("window shape cannot be larger than input array
        # shape"). A feature can be varying overall and constant in one fold,
        # so this has to happen per fit, not once during selection.
        varying = X.nunique(dropna=False) > 1
        if not varying.all():
            dropped = list(X.columns[~varying])
            logger.info(
                "dropping %d constant feature(s) in this training window: %s",
                len(dropped), ", ".join(dropped[:5]) + ("..." if len(dropped) > 5 else ""),
            )
            X = X.loc[:, varying]
        if X.shape[1] == 0:
            raise ValueError(
                "every feature is constant across this training window — "
                "the window is too short or the data is degenerate"
            )

        self.feature_names_ = list(X.columns)
        y_arr = y.to_numpy().astype(int)
        w_arr = sample_weight.to_numpy() if sample_weight is not None else None

        folds = self._internal_folds(dates, t1)
        last_train, last_test = folds[-1]
        rng = np.random.default_rng(self._seed)
        report = FitReport(n_fit=len(X))

        # ---- 1) hyperparameter search on the last (largest) internal fold --
        X_lt, y_lt = X.iloc[last_train], y_arr[last_train]
        w_lt = w_arr[last_train] if w_arr is not None else None
        X_lv, y_lv = X.iloc[last_test], y_arr[last_test]
        chosen: dict[str, dict[str, Any]] = {}
        last_fold_model: dict[str, Any] = {}
        for name in self._cfg.members:
            candidates = [_default_params(name)] + [
                _sample_params(name, rng) for _ in range(self._cfg.tuning_iterations)
            ]
            best: tuple[float, dict[str, Any], Any] | None = None
            for params in candidates:
                model = _build_member(name, params, self._seed)
                _fit_member(model, X_lt, y_lt, w_lt)
                p = model.predict_proba(X_lv)[:, 1]
                score = log_loss(y_lv, np.clip(p, 1e-6, 1 - 1e-6))
                if best is None or score < best[0]:
                    best = (score, params, model)
            assert best is not None
            chosen[name] = best[1]
            last_fold_model[name] = best[2]  # reuse as the OOF fit for the last fold

        # ---- 2) out-of-fold predictions across all internal folds ----------
        oof_probs: dict[str, list[np.ndarray]] = {n: [] for n in self._cfg.members}
        oof_y: list[np.ndarray] = []
        for train_idx, test_idx in folds:
            X_te = X.iloc[test_idx]
            oof_y.append(y_arr[test_idx])
            is_last = test_idx is last_test
            for name in self._cfg.members:
                if is_last:
                    model = last_fold_model[name]
                else:
                    model = _build_member(name, chosen[name], self._seed)
                    _fit_member(
                        model,
                        X.iloc[train_idx],
                        y_arr[train_idx],
                        w_arr[train_idx] if w_arr is not None else None,
                    )
                oof_probs[name].append(model.predict_proba(X_te)[:, 1])

        y_oof = np.concatenate(oof_y)
        member_oof = {n: np.concatenate(ps) for n, ps in oof_probs.items()}
        report.n_calib = len(y_oof)

        for name in self._cfg.members:
            p = member_oof[name]
            loss = float(log_loss(y_oof, np.clip(p, 1e-6, 1 - 1e-6)))
            auc = float(roc_auc_score(y_oof, p)) if len(np.unique(y_oof)) > 1 else float("nan")
            report.members.append(
                MemberReport(name=name, params=chosen[name], calib_log_loss=loss,
                             calib_auc=auc, weight=0.0)
            )

        losses = np.array([m.calib_log_loss for m in report.members])
        raw_w = np.exp(-(losses - losses.min()) / _WEIGHT_TEMPERATURE)
        weights = raw_w / raw_w.sum()
        for m, w in zip(report.members, weights):
            m.weight = float(w)
            self._weights[m.name] = float(w)

        # ---- 3) calibrate on pooled OOF ensemble probabilities -------------
        p_ens = np.sum(
            [self._weights[n] * member_oof[n] for n in self._cfg.members], axis=0
        )
        self._fit_calibrator(p_ens, y_oof, report)
        p_final = self._apply_calibrator(p_ens)
        report.calib_brier = float(np.mean((p_final - y_oof) ** 2))
        # Kept in full for Venn-ABERS intervals: the band must be computed on
        # exactly the evidence the calibrator saw, which guarantees (isotonic
        # monotonicity under single-point augmentation) that p0 <= p̂ <= p1 —
        # a subsample here made bands that excluded their own point estimate.
        self._oof_scores = p_ens.astype(float)
        self._oof_labels = y_oof.astype(int)

        # ---- 4) refit members on the FULL training window ------------------
        for name in self._cfg.members:
            model = _build_member(name, chosen[name], self._seed)
            _fit_member(model, X, y_arr, w_arr)
            self._members[name] = model

        self.report_ = report
        logger.info(
            "ensemble fit (K=%d OOF, %d rows pooled): %s | oof brier=%.4f",
            len(folds), len(y_oof),
            {m.name: round(m.weight, 2) for m in report.members},
            report.calib_brier,
        )
        return self

    def _fit_calibrator(self, p: np.ndarray, y: np.ndarray, report: FitReport) -> None:
        kind = self._cfg.calibration
        if kind == "isotonic" and len(y) < _MIN_CALIB_ROWS_ISOTONIC:
            kind = "sigmoid"  # isotonic overfits tiny calibration sets
        if kind == "isotonic":
            cal = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
            cal.fit(p, y)
        else:
            logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
            cal = LogisticRegression(C=1e6, max_iter=1000)
            cal.fit(logit.reshape(-1, 1), y)
        self._calibrator = cal
        self._calibration_kind = kind
        report.calibration = kind

    def _apply_calibrator(self, p: np.ndarray) -> np.ndarray:
        if self._calibration_kind == "isotonic":
            return np.asarray(self._calibrator.predict(p))
        logit = np.log(np.clip(p, 1e-6, 1 - 1e-6) / np.clip(1 - p, 1e-6, 1 - 1e-6))
        return self._calibrator.predict_proba(logit.reshape(-1, 1))[:, 1]

    # ------------------------------------------------------------------ #

    def _raw_member_matrix(self, X: pd.DataFrame) -> np.ndarray:
        X = X.reindex(columns=self.feature_names_)
        return np.column_stack([m.predict_proba(X)[:, 1] for m in self._members.values()])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated (n, 2) probability matrix, sklearn-compatible."""
        probs = self._raw_member_matrix(X)
        w = np.array([self._weights[n] for n in self._members])
        p_raw = probs @ w
        p = np.clip(self._apply_calibrator(p_raw), 1e-4, 1 - 1e-4)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def uncertainty(self, X: pd.DataFrame) -> np.ndarray:
        """Member disagreement: std of member probabilities per row."""
        return self._raw_member_matrix(X).std(axis=1)

    @property
    def has_intervals(self) -> bool:
        """False for bundles saved before Venn-ABERS support existed."""
        return getattr(self, "_oof_scores", None) is not None

    def raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """Pre-calibration weighted member blend — the Venn-ABERS score axis.

        Batch this once per prediction frame; per-row member predictions are
        two orders of magnitude slower than one vectorized pass.
        """
        probs = self._raw_member_matrix(X)
        w = np.array([self._weights[n] for n in self._members])
        return np.asarray(probs @ w, dtype=float)

    def interval_for_scores(self, scores: np.ndarray) -> np.ndarray:
        """(n, 2) Venn-ABERS band [p0, p1] for precomputed raw scores."""
        if not self.has_intervals:
            raise RuntimeError("fit the ensemble before requesting intervals")
        assert self._oof_scores is not None and self._oof_labels is not None
        out = np.empty((len(scores), 2), dtype=float)
        for i, s in enumerate(scores):
            out[i] = venn_abers_interval(self._oof_scores, self._oof_labels, float(s))
        return out

    def probability_interval(self, X: pd.DataFrame) -> np.ndarray:
        """(n, 2) inductive Venn-ABERS band [p0, p1] per row.

        Computed on the raw (pre-calibrator) ensemble score against the
        pooled OOF calibration sample — the same evidence the isotonic
        calibrator saw, so the band brackets what that calibration can
        legitimately claim at this score.
        """
        if not self.has_intervals:
            raise RuntimeError("fit the ensemble before requesting intervals")
        return self.interval_for_scores(self.raw_scores(X))
