"""Unsupervised market regime detection with deterministic refinement.

Base states come from a Gaussian mixture over standardized (trend, volatility,
drawdown) of the benchmark — unsupervised, so the model discovers the market's
own clustering rather than inheriting our priors. Components are then mapped
to base labels (bull / range / turbulent / bear) by their moments, posteriors
are made *sticky* with a causal EWMA (regimes are persistent; bar-by-bar
flip-flopping is noise), and deterministic rules refine the base state into
the full taxonomy (strong/weak bull, accumulation/distribution, correction,
crash) plus a volatility overlay.

Causality: ``fit`` may only see the training window; ``transform`` uses frozen
mixture parameters and *trailing* statistics (rolling percentiles, trailing
drawdown, causal EWMA), so the value at ``t`` never depends on data after
``t``. This is verified by tests.

Regime output gates strategy activation and scales risk. It is deliberately
NOT fed to the ML ensemble as a feature: the panel already carries the
underlying trend/vol/breadth information, and keeping the detector out of the
feature path means a regime misclassification cannot silently poison the
probability model as well — defense in depth.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from titan.core.config import RegimeConfig
from titan.core.timeframe import TRADING_DAYS_PER_YEAR as TRADING_DAYS
from titan.core.types import Regime, VolState
from titan.features.rolling import realized_vol, rolling_percentile_rank, rolling_slope_stats


@dataclass(slots=True)
class RegimeSnapshot:
    regime: Regime
    vol_state: VolState
    confidence: float
    posteriors: dict[str, float]

    def to_dict(self) -> dict:
        return {
            "regime": self.regime.value,
            "vol_state": self.vol_state.value,
            "confidence": round(self.confidence, 4),
            "posteriors": {k: round(v, 4) for k, v in self.posteriors.items()},
        }


BASE_STATES = ("bull", "range", "turbulent", "bear")


class RegimeDetector:
    """GMM base states + sticky smoothing + rule refinement."""

    def __init__(
        self, cfg: RegimeConfig, seed: int = 7, periods_per_year: float = TRADING_DAYS
    ) -> None:
        self._cfg = cfg
        self._seed = seed
        # Both the annualization of the trend feature and the "past year"
        # lookbacks below are stated in years, so they must be expressed in
        # whatever bar this run uses.
        self._periods_per_year = periods_per_year
        self._year_bars = max(round(periods_per_year), 2)
        self._gmm: GaussianMixture | None = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._component_to_base: dict[int, str] = {}

    # ------------------------------------------------------------------ #

    def _state_features(self, bench: pd.DataFrame) -> pd.DataFrame:
        close = bench["close"]
        trend = np.log(close).diff(self._cfg.trend_window)
        vol = realized_vol(close, self._cfg.vol_window, annualize=self._periods_per_year)
        # Rolling (not all-time) drawdown: an old crash must not keep tainting
        # the state estimate a year later.
        dd = close / close.rolling(
            self._year_bars, min_periods=max(self._year_bars // 4, 2)
        ).max() - 1.0
        slope_t = rolling_slope_stats(np.log(close), self._cfg.trend_window)[1]
        return pd.DataFrame({"trend": trend, "vol": vol, "dd": dd, "slope_t": slope_t})

    def fit(self, bench: pd.DataFrame) -> RegimeDetector:
        # Cluster on current *dynamics* (trend, vol). Drawdown is deliberately
        # excluded: it is path-dependent and smears cluster boundaries; it
        # enters only through the deterministic refinement rules.
        feats = self._state_features(bench)[["trend", "vol"]].dropna()
        if len(feats) < self._cfg.min_train_bars:
            raise ValueError(f"need >= {self._cfg.min_train_bars} bars to fit regimes")
        values = feats.to_numpy()
        self._mu = values.mean(axis=0)
        self._sd = values.std(axis=0) + 1e-12
        z = (values - self._mu) / self._sd
        gmm = GaussianMixture(
            n_components=self._cfg.n_states,
            covariance_type="full",
            random_state=self._seed,
            reg_covar=1e-4,
            n_init=3,
        )
        gmm.fit(z)
        self._gmm = gmm
        self._component_to_base = self._map_components(gmm)
        return self

    def _map_components(self, gmm: GaussianMixture) -> dict[int, str]:
        """Label mixture components by their *economic* moments.

        Component means are de-standardized back to raw units and labelled by
        annualized trend and relative volatility. Threshold-based, not
        rank-based: several components may describe the same economic state
        (e.g. two flavours of bull), and forcing exactly one component per
        label misassigns whole stretches of history.
        """
        assert self._mu is not None and self._sd is not None
        raw_means = gmm.means_ * self._sd + self._mu  # (trend_63, ann_vol)
        ann_factor = self._periods_per_year / self._cfg.trend_window
        median_vol = float(self._mu[1])
        mapping: dict[int, str] = {}
        for comp in range(raw_means.shape[0]):
            ann_trend = raw_means[comp, 0] * ann_factor
            vol = raw_means[comp, 1]
            if ann_trend <= -0.10:
                mapping[comp] = "bear"
            elif ann_trend >= 0.07:
                mapping[comp] = "turbulent" if vol >= 2.0 * median_vol else "bull"
            elif vol >= 1.5 * median_vol:
                mapping[comp] = "turbulent"
            else:
                mapping[comp] = "range"
        # Degenerate fits (all components one label) keep extremes distinct.
        if len(set(mapping.values())) == 1:
            order = np.argsort(raw_means[:, 0])
            mapping[int(order[0])] = "bear"
            mapping[int(order[-1])] = "bull"
        return mapping

    # ------------------------------------------------------------------ #

    def transform(self, bench: pd.DataFrame) -> pd.DataFrame:
        """Per-bar regime classification using frozen mixture parameters."""
        if self._gmm is None or self._mu is None or self._sd is None:
            raise RuntimeError("RegimeDetector must be fit before transform")

        feats = self._state_features(bench)
        valid = feats[["trend", "vol"]].notna().all(axis=1)
        z = (feats.loc[valid, ["trend", "vol"]].to_numpy() - self._mu) / self._sd
        raw_post = self._gmm.predict_proba(z)

        # Aggregate component posteriors into base-state posteriors.
        base_post = pd.DataFrame(0.0, index=feats.index[valid], columns=list(BASE_STATES))
        for comp, base in self._component_to_base.items():
            base_post[base] += raw_post[:, comp]

        # Sticky causal smoothing.
        smoothed = base_post.ewm(halflife=self._cfg.smoothing_halflife, adjust=False).mean()
        smoothed = smoothed.div(smoothed.sum(axis=1), axis=0)

        vol_pct = rolling_percentile_rank(feats["vol"], self._year_bars).reindex(feats.index)
        dd = feats["dd"]
        slope_t = feats["slope_t"]

        out = pd.DataFrame(index=feats.index, columns=["regime", "vol_state"], dtype=object)
        out["confidence"] = np.nan
        for base in BASE_STATES:
            out[f"p_{base}"] = smoothed[base].reindex(feats.index)

        trend = feats["trend"]
        for ts in smoothed.index:
            base = str(smoothed.loc[ts].idxmax())
            out.loc[ts, "regime"] = self._refine(
                base,
                vol_pct=float(vol_pct.get(ts, np.nan)),
                dd=float(dd.get(ts, np.nan)),
                slope_t=float(slope_t.get(ts, np.nan)),
                trend=float(trend.get(ts, np.nan)),
            ).value
            out.loc[ts, "confidence"] = float(smoothed.loc[ts].max())
            out.loc[ts, "vol_state"] = self._vol_state(float(vol_pct.get(ts, np.nan))).value
        return out.dropna(subset=["regime"])

    # ------------------------------------------------------------------ #

    def _refine(self, base: str, vol_pct: float, dd: float, slope_t: float, trend: float) -> Regime:
        cfg = self._cfg
        crash = (
            not np.isnan(vol_pct)
            and vol_pct >= cfg.crash_vol_percentile
            and not np.isnan(dd)
            and dd <= cfg.crash_drawdown
        )
        if crash:
            return Regime.CRASH
        if base == "bear":
            return Regime.BEAR
        if base == "bull":
            # A correction is an ACTIVE drawdown: below the threshold with a
            # negative trailing return. Recovering from old lows is not one.
            if not np.isnan(dd) and dd <= cfg.correction_drawdown and trend < 0.0:
                return Regime.CORRECTION
            if slope_t >= 2.0:
                return Regime.STRONG_BULL
            if slope_t >= 0.5:
                return Regime.BULL
            return Regime.WEAK_BULL
        if base == "range":
            if slope_t >= 1.5:
                return Regime.ACCUMULATION
            if slope_t <= -1.5:
                return Regime.DISTRIBUTION
            return Regime.RANGE
        # turbulent: a transition state; direction decides the lean.
        if slope_t >= 0.5:
            return Regime.WEAK_BULL
        if slope_t <= -0.5:
            return Regime.BEAR
        return Regime.RANGE

    @staticmethod
    def _vol_state(vol_pct: float) -> VolState:
        if np.isnan(vol_pct):
            return VolState.NORMAL
        if vol_pct < 0.25:
            return VolState.LOW
        if vol_pct < 0.75:
            return VolState.NORMAL
        if vol_pct < 0.95:
            return VolState.HIGH
        return VolState.EXTREME

    def snapshot(self, bench: pd.DataFrame) -> RegimeSnapshot:
        """Latest-bar regime state."""
        table = self.transform(bench)
        row = table.iloc[-1]
        return RegimeSnapshot(
            regime=Regime(row["regime"]),
            vol_state=VolState(row["vol_state"]),
            confidence=float(row["confidence"]),
            posteriors={b: float(row[f"p_{b}"]) for b in BASE_STATES},
        )
