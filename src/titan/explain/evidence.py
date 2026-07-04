"""Model-agnostic local explanations by median-counterfactual.

For each of the globally important features we ask the fitted ensemble one
concrete question: *"how would the probability change if this feature sat at
its training median instead of its current value?"* The signed difference is
that feature's local impact. This is an ICE-style counterfactual — honest,
model-agnostic and cheap (one batched ``predict_proba`` per explanation) —
and it requires no assumptions about feature independence beyond what any
perturbation method needs.

The explainer also carries the training ECDF of every feature, so evidence
strings can say "vol-adjusted momentum in the 92nd percentile of its history"
instead of quoting raw numbers nobody can interpret.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_ECDF_SAMPLE = 512


@dataclass(slots=True)
class FeatureEffect:
    feature: str
    value: float
    percentile: float  # of the training distribution, in [0, 1]
    delta_p: float     # p(actual) - p(feature at median): positive supports long

    def to_dict(self) -> dict:
        return {
            "feature": self.feature,
            "value": None if np.isnan(self.value) else round(self.value, 5),
            "percentile": round(self.percentile, 3),
            "delta_p": round(self.delta_p, 5),
        }


class LocalExplainer:
    def __init__(self, model, X_train: pd.DataFrame, top_features: list[str], seed: int = 7) -> None:
        """
        Parameters
        ----------
        model:
            Anything with ``predict_proba(DataFrame) -> (n, 2)``.
        X_train:
            Training feature frame (selected columns).
        top_features:
            Globally important features to probe, importance-ordered.
        """
        self._model = model
        self._features = [f for f in top_features if f in X_train.columns]
        self._medians = X_train.median()
        rng = np.random.default_rng(seed)
        take = min(_ECDF_SAMPLE, len(X_train))
        rows = rng.choice(len(X_train), size=take, replace=False)
        self._ecdf_sample = {
            col: np.sort(X_train[col].iloc[rows].dropna().to_numpy()) for col in X_train.columns
        }
        self._columns = list(X_train.columns)

    def percentile(self, feature: str, value: float) -> float:
        sample = self._ecdf_sample.get(feature)
        if sample is None or len(sample) == 0 or np.isnan(value):
            return 0.5
        return float(np.searchsorted(sample, value, side="right") / len(sample))

    def explain(self, x_row: pd.Series, max_features: int | None = None) -> list[FeatureEffect]:
        feats = self._features[: max_features or len(self._features)]
        if not feats:
            return []
        base = x_row.reindex(self._columns)
        batch = pd.DataFrame([base] * (len(feats) + 1))
        for i, f in enumerate(feats, start=1):
            batch.iloc[i, batch.columns.get_loc(f)] = self._medians[f]
        probs = self._model.predict_proba(batch)[:, 1]
        p_actual = float(probs[0])
        effects = [
            FeatureEffect(
                feature=f,
                value=float(base[f]) if not pd.isna(base[f]) else float("nan"),
                percentile=self.percentile(f, float(base[f]) if not pd.isna(base[f]) else np.nan),
                delta_p=p_actual - float(probs[i]),
            )
            for i, f in enumerate(feats, start=1)
        ]
        effects.sort(key=lambda e: abs(e.delta_p), reverse=True)
        return effects
