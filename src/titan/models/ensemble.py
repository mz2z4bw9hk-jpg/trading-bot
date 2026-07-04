"""Calibrated heterogeneous ensemble with honest internal validation.

Members (gradient boosting, random forest, regularized logistic regression)
are diverse by construction: different bias structures disagree in different
ways, and the disagreement itself is a usable uncertainty estimate.

Fitting protocol (all inside the *training* window of an outer fold — the
outer walk-forward never sees any of this):

1. Split train into fit / calibration parts by date, separated by a purge gap
   of one label horizon so calibration events cannot overlap fit events.
2. Random-search each member's hyperparameters on the fit part, scored on the
   calibration part (random search is a strong baseline — Bergstra & Bengio
   2012; the interface accepts any sampler, so Bayesian optimization can be
   plugged in without touching callers).
3. Weight members by exponentiated negative calibration log-loss.
4. Fit an isotonic (or Platt) calibrator on the weighted ensemble's
   calibration-part probabilities. Raw scores of boosted trees are *not*
   probabilities; position sizing needs calibrated ones.

``predict_proba`` returns calibrated P(take-profit before stop); ``uncertainty``
returns member disagreement (std of member probabilities).
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

_CALIB_FRACTION = 0.2
_MIN_CALIB_ROWS_ISOTONIC = 300
_WEIGHT_TEMPERATURE = 0.02  # log-loss units; smaller = sharper member weighting


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
    return {
        "hgb": {"max_iter": 200, "learning_rate": 0.08, "max_leaf_nodes": 31,
                "min_samples_leaf": 50, "l2_regularization": 0.1},
        "rf": {"n_estimators": 200, "max_depth": 6, "min_samples_leaf": 50, "max_features": "sqrt"},
        "logistic": {"C": 1.0},
    }[member]


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
        self.feature_names_: list[str] = []
        self.report_: FitReport | None = None
        self.classes_ = np.array([0, 1])

    # ------------------------------------------------------------------ #

    def _split_fit_calib(self, dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
        values = to_naive_utc(dates)
        unique = np.unique(values)
        cut_pos = int(len(unique) * (1.0 - _CALIB_FRACTION))
        cut_pos = min(max(cut_pos, 1), len(unique) - 2)
        calib_start = unique[min(cut_pos + self._horizon, len(unique) - 1)]
        fit_end = unique[cut_pos - 1]
        fit_mask = values <= fit_end
        calib_mask = values >= calib_start
        return np.where(fit_mask)[0], np.where(calib_mask)[0]

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        dates: pd.DatetimeIndex,
        sample_weight: pd.Series | None = None,
    ) -> CalibratedEnsemble:
        if len(X) != len(y) or len(X) != len(dates):
            raise ValueError("X, y, dates must align")
        if len(X) > self._cfg.max_train_rows:
            X, y = X.iloc[-self._cfg.max_train_rows :], y.iloc[-self._cfg.max_train_rows :]
            dates = dates[-self._cfg.max_train_rows :]
            if sample_weight is not None:
                sample_weight = sample_weight.iloc[-self._cfg.max_train_rows :]

        self.feature_names_ = list(X.columns)
        y_arr = y.to_numpy().astype(int)
        w_arr = sample_weight.to_numpy() if sample_weight is not None else None

        fit_idx, calib_idx = self._split_fit_calib(dates)
        if len(fit_idx) < 200 or len(calib_idx) < 50:
            raise ValueError(
                f"training window too small: fit={len(fit_idx)}, calib={len(calib_idx)}"
            )
        X_fit, y_fit = X.iloc[fit_idx], y_arr[fit_idx]
        X_cal, y_cal = X.iloc[calib_idx], y_arr[calib_idx]
        w_fit = w_arr[fit_idx] if w_arr is not None else None

        rng = np.random.default_rng(self._seed)
        report = FitReport(n_fit=len(fit_idx), n_calib=len(calib_idx))

        member_probs: dict[str, np.ndarray] = {}
        for name in self._cfg.members:
            candidates = [_default_params(name)] + [
                _sample_params(name, rng) for _ in range(self._cfg.tuning_iterations)
            ]
            best: tuple[float, dict[str, Any], Any] | None = None
            for params in candidates:
                model = _build_member(name, params, self._seed)
                _fit_member(model, X_fit, y_fit, w_fit)
                p = model.predict_proba(X_cal)[:, 1]
                score = log_loss(y_cal, np.clip(p, 1e-6, 1 - 1e-6))
                if best is None or score < best[0]:
                    best = (score, params, model)
            assert best is not None
            loss, params, model = best
            self._members[name] = model
            p_cal = model.predict_proba(X_cal)[:, 1]
            member_probs[name] = p_cal
            auc = roc_auc_score(y_cal, p_cal) if len(np.unique(y_cal)) > 1 else float("nan")
            report.members.append(
                MemberReport(name=name, params=params, calib_log_loss=loss,
                             calib_auc=float(auc), weight=0.0)
            )

        losses = np.array([m.calib_log_loss for m in report.members])
        raw_w = np.exp(-(losses - losses.min()) / _WEIGHT_TEMPERATURE)
        weights = raw_w / raw_w.sum()
        for m, w in zip(report.members, weights):
            m.weight = float(w)
            self._weights[m.name] = float(w)

        p_ens = np.sum(
            [self._weights[n] * member_probs[n] for n in self._members], axis=0
        )
        self._fit_calibrator(p_ens, y_cal, report)
        p_final = self._apply_calibrator(p_ens)
        report.calib_brier = float(np.mean((p_final - y_cal) ** 2))
        self.report_ = report
        logger.info(
            "ensemble fit: %s | brier=%.4f",
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
