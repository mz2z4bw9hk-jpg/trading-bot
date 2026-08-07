"""Walk-forward research orchestration.

For every fold, strictly in causal order:

1. feature screening (univariate IC) → redundancy pruning → top-K selection,
   all on TRAINING data only;
2. calibrated ensemble fit (with purged internal calibration split);
3. regime detector fit on the training window of the benchmark, then frozen
   and rolled causally across the test window;
4. analogue index + local explainer fit on training data;
5. signal generation over the TEST window through the adaptive EV gate;
6. risk-engine-approved trade plans.

All folds' plans are then executed in ONE portfolio simulation over the
stitched out-of-sample period, followed by block-bootstrap confidence
intervals, deflated Sharpe sensitivity, and per-regime attribution. Nothing
in the report ever saw its own training data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sstats
from sklearn.metrics import log_loss, roc_auc_score

from titan.backtest.costs import CostModel
from titan.backtest.engine import BacktestEngine, BacktestResult, TradePlan
from titan.backtest.metrics import deflated_sharpe_ratio
from titan.backtest.monte_carlo import BootstrapReport, bootstrap_analysis, risk_of_ruin
from titan.core.config import TitanConfig, bar_clock
from titan.core.log import get_logger
from titan.core.timeframe import TRADING_DAYS_PER_YEAR as TRADING_DAYS
from titan.core.timeframe import warn_on_calendar_mismatch
from titan.core.types import Regime, VolState
from titan.data.store import MarketDataset
from titan.explain.evidence import LocalExplainer
from titan.features.pipeline import FeatureMatrixBuilder
from titan.features.registry import FeaturePanel
from titan.features.selection import permutation_rank, redundancy_prune, univariate_ic
from titan.labels.triple_barrier import build_label_panel
from titan.models.cv import PurgedWalkForward, assert_no_leakage
from titan.models.ensemble import CalibratedEnsemble
from titan.regime.detector import RegimeDetector
from titan.risk.portfolio import RiskEngine, build_returns_matrix
from titan.signals.analogues import AnalogueIndex
from titan.signals.generator import SignalGenerator
from titan.signals.schema import Signal

logger = get_logger(__name__)


def _lazy_interval(
    ensemble: CalibratedEnsemble, raw_score: float
) -> Callable[[], tuple[float, float]]:
    """Venn-ABERS band for one candidate, deferred until the cheap gates pass.

    Takes the candidate's precomputed raw ensemble score (batched once per
    fold) so the deferred work is only the two isotonic fits.
    """

    def provide() -> tuple[float, float]:
        band = ensemble.interval_for_scores(np.array([raw_score]))[0]
        return float(band[0]), float(band[1])

    return provide


@dataclass(slots=True)
class FoldReport:
    fold: int
    test_start: str
    test_end: str
    n_train: int
    n_test: int
    n_features: int
    auc: float
    brier: float
    log_loss_: float
    n_signals: int
    member_weights: dict[str, float] = field(default_factory=dict)
    selected_features: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "fold": self.fold,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_features": self.n_features,
            "auc": round(self.auc, 4),
            "brier": round(self.brier, 5),
            "log_loss": round(self.log_loss_, 5),
            "n_signals": self.n_signals,
            "member_weights": {k: round(v, 3) for k, v in self.member_weights.items()},
            "selected_features": self.selected_features,
        }


@dataclass(slots=True)
class WalkForwardReport:
    folds: list[FoldReport]
    pooled_auc: float
    pooled_brier: float
    high_conf_hit_rate: float
    base_rate: float
    backtest: BacktestResult
    bootstrap: BootstrapReport
    dsr_sensitivity: dict[str, float]
    ruin: dict
    regime_breakdown: dict[str, dict]
    signals: list[Signal]
    importance: pd.Series | None
    regimes: pd.DataFrame
    decisions: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "folds": [f.to_dict() for f in self.folds],
            "pooled_auc": round(self.pooled_auc, 4),
            "pooled_brier": round(self.pooled_brier, 5),
            "high_conf_hit_rate": round(self.high_conf_hit_rate, 4),
            "base_rate": round(self.base_rate, 4),
            "backtest": self.backtest.to_dict(),
            "bootstrap": self.bootstrap.to_dict(),
            "dsr_sensitivity": {k: round(v, 4) for k, v in self.dsr_sensitivity.items()},
            "risk_of_ruin": self.ruin,
            "regime_breakdown": self.regime_breakdown,
            "n_signals": len(self.signals),
            "importance_top20": (
                {k: round(float(v), 6) for k, v in self.importance.head(20).items()}
                if self.importance is not None
                else {}
            ),
        }


class WalkForwardRunner:
    def __init__(self, cfg: TitanConfig) -> None:
        self._cfg = cfg
        # Final-fold artifacts (ensemble/selected/explainer/analogues), stashed
        # by run() so the scanner can reuse the freshest model without refitting.
        self.last_fold_artifacts: dict[str, Any] = {}

    # ------------------------------------------------------------------ #

    def _select_features(
        self, X_train: pd.DataFrame, y_train: pd.Series, seed: int
    ) -> list[str]:
        ic = univariate_ic(X_train, y_train, seed=seed)
        prune = redundancy_prune(
            X_train,
            threshold=self._cfg.features.redundancy_threshold,
            priority=ic.abs(),
            seed=seed,
        )
        ranked = ic.abs()[prune.kept].sort_values(ascending=False)
        return list(ranked.head(self._cfg.features.max_features).index)

    # ------------------------------------------------------------------ #

    def run(self, dataset: MarketDataset, panel: FeaturePanel | None = None) -> WalkForwardReport:
        cfg = self._cfg
        seed = cfg.run.seed

        if panel is None:
            panel = FeatureMatrixBuilder(cfg.features).build(dataset)
        labels, weights = build_label_panel(dataset.frames, cfg.labels)
        common = panel.X.index.intersection(labels.index)
        X_all = panel.X.loc[common]
        lab = labels.loc[common]
        w_all = weights.loc[common]
        reliability = (
            panel.reliability.loc[common]
            if panel.reliability is not None
            else pd.Series(1.0, index=common)
        )
        dates = pd.DatetimeIndex(X_all.index.get_level_values(0))

        splitter = PurgedWalkForward(cfg.cv)
        folds = splitter.split(dates, lab["t1"])
        assert_no_leakage(folds, dates, lab["t1"])
        logger.info("walk-forward: %d folds over %d panel rows", len(folds), len(X_all))

        cost_model = CostModel(cfg.backtest.costs)
        clock = bar_clock(cfg)
        warn_on_calendar_mismatch(clock, dataset.benchmark_frame.index)
        ppy = clock.bars_per_year
        logger.info(
            "bar clock: %s, %.0f bars/year (%s calendar)",
            clock.timeframe, ppy, "24/7" if clock.continuous else "session",
        )
        generator = SignalGenerator(
            cfg.signals, cfg.labels, cfg.risk, cost_model, cfg.backtest.max_positions,
            periods_per_year=ppy,
        )

        # Precomputed causal per-symbol volatility (cheap candidate pre-screen).
        sigma_ctx: dict[str, pd.Series] = {}
        for sym, frame in dataset.frames.items():
            sigma_ctx[sym] = (
                np.log(frame["close"]).diff()
                .ewm(span=cfg.labels.vol_span, adjust=False, min_periods=cfg.labels.vol_span)
                .std()
            )

        fold_reports: list[FoldReport] = []
        all_signals: list[Signal] = []
        plans: list[TradePlan] = []
        pooled_p: list[np.ndarray] = []
        pooled_y: list[np.ndarray] = []
        decision_rows: list[dict] = []
        regime_tables: list[pd.DataFrame] = []
        importance: pd.Series | None = None
        last_ensemble: CalibratedEnsemble | None = None
        last_selected: list[str] = []
        last_explainer: LocalExplainer | None = None
        last_analogues: AnalogueIndex | None = None

        for fold in folds:
            Xtr = X_all.iloc[fold.train_idx]
            ytr = lab["label"].iloc[fold.train_idx]
            wtr = w_all.iloc[fold.train_idx]
            Xte = X_all.iloc[fold.test_idx]
            yte = lab["label"].iloc[fold.test_idx]

            selected = self._select_features(Xtr, ytr, seed)
            ensemble = CalibratedEnsemble(cfg.model, cfg.labels.horizon_bars, seed=seed)
            ensemble.fit(
                Xtr[selected], ytr, dates[fold.train_idx], wtr,
                t1=lab["t1"].iloc[fold.train_idx],
            )

            p_te = ensemble.predict_proba(Xte[selected])[:, 1]
            unc_te = ensemble.uncertainty(Xte[selected])
            raw_te = ensemble.raw_scores(Xte[selected])
            auc = float(roc_auc_score(yte, p_te)) if yte.nunique() > 1 else float("nan")
            brier = float(np.mean((p_te - yte.to_numpy()) ** 2))
            ll = float(log_loss(yte, np.clip(p_te, 1e-6, 1 - 1e-6)))
            pooled_p.append(p_te)
            pooled_y.append(yte.to_numpy())

            # Regime: fit on the training window of the benchmark, frozen roll.
            bench = dataset.benchmark_frame
            bench_train = bench.loc[: fold.train_end]
            detector = RegimeDetector(
                cfg.regime, seed=seed, periods_per_year=ppy
            ).fit(bench_train)
            regime_table = detector.transform(bench.loc[: fold.test_end])
            regime_test = regime_table.loc[fold.test_start :]
            regime_tables.append(regime_test)

            # Analogues + explainer, training data only.
            outcomes = lab.iloc[fold.train_idx][["label", "ret", "bars_held", "mae", "mfe"]]
            analogues = AnalogueIndex(k=cfg.signals.analogue_k, seed=seed).fit(
                Xtr[selected], outcomes
            )
            ic_order = (
                univariate_ic(Xtr[selected], ytr, seed=seed).abs()
                .sort_values(ascending=False)
            )
            explainer = LocalExplainer(ensemble, Xtr[selected], list(ic_order.index), seed=seed)

            # Signal generation over the test window.
            n_signals_fold = 0
            test_index = Xte.index
            p_series = pd.Series(p_te, index=test_index)
            unc_series = pd.Series(unc_te, index=test_index)
            raw_series = pd.Series(raw_te, index=test_index)
            for (ts, sym) in test_index:
                p = float(p_series[(ts, sym)])
                decision_rows.append(
                    {
                        "date": ts,
                        "symbol": sym,
                        "p": p,
                        "uncertainty": float(unc_series[(ts, sym)]),
                        "label": int(lab.at[(ts, sym), "label"]),
                        "fold": fold.fold,
                    }
                )
                if p < cfg.signals.min_probability:  # cheap pre-gate
                    continue
                if ts not in regime_table.index:
                    continue
                reg_row = regime_table.loc[ts]
                sigma_v = sigma_ctx[sym].get(ts, np.nan)
                if not np.isfinite(sigma_v):
                    continue
                frame_slice = dataset.frames[sym].loc[:ts]
                signal = generator.generate(
                    symbol=sym,
                    date=ts,
                    probability=p,
                    uncertainty=float(unc_series[(ts, sym)]),
                    frame=frame_slice.tail(60),
                    feature_row=Xte.loc[(ts, sym), selected],
                    regime=Regime(reg_row["regime"]),
                    vol_state=VolState(reg_row["vol_state"]),
                    regime_confidence=float(reg_row["confidence"]),
                    analogue=analogues.query(Xte.loc[(ts, sym), selected]),
                    reliability=float(reliability.get((ts, sym), 1.0)),
                    explainer=explainer,
                    model_version=f"wf_fold{fold.fold}",
                    interval_provider=_lazy_interval(ensemble, float(raw_series[(ts, sym)])),
                )
                if signal is None:
                    continue
                all_signals.append(signal)
                n_signals_fold += 1
                plans.append(
                    TradePlan(
                        symbol=sym,
                        decision_date=ts,
                        size_fraction=signal.position_size_fraction,
                        stop_price=signal.stop_loss,
                        tp_price=signal.take_profit_levels[1],
                        max_holding_bars=cfg.labels.horizon_bars,
                        entry_ref=signal.market_entry,
                        priority=signal.confidence_score,
                        tag=f"{signal.trade_grade.value}|f{fold.fold}",
                    )
                )

            fold_reports.append(
                FoldReport(
                    fold=fold.fold,
                    test_start=str(fold.test_start.date()),
                    test_end=str(fold.test_end.date()),
                    n_train=len(fold.train_idx),
                    n_test=len(fold.test_idx),
                    n_features=len(selected),
                    auc=auc,
                    brier=brier,
                    log_loss_=ll,
                    n_signals=n_signals_fold,
                    member_weights=(
                        {m.name: m.weight for m in ensemble.report_.members}
                        if ensemble.report_
                        else {}
                    ),
                    selected_features=selected,
                )
            )
            logger.info(
                "fold %d: auc=%.4f brier=%.4f signals=%d (features=%d)",
                fold.fold, auc, brier, n_signals_fold, len(selected),
            )
            last_ensemble, last_selected = ensemble, selected
            last_explainer, last_analogues = explainer, analogues

        # ---- OOS portfolio simulation --------------------------------------
        regimes = pd.concat(regime_tables).sort_index()
        regimes = regimes[~regimes.index.duplicated(keep="first")]
        returns_wide = build_returns_matrix(dataset.frames, cfg.data.timeframe)
        risk_engine = RiskEngine(
            cfg.risk,
            universe=dataset.universe,
            returns=returns_wide,
            regimes=regimes["regime"],
            var_window_bars=max(round(ppy), 2),
        )
        engine = BacktestEngine(
            cfg.backtest, cost_model, universe=dataset.universe, risk_approver=risk_engine,
            periods_per_year=ppy,
        )
        oos_start = folds[0].test_start
        result = engine.run(dataset.frames, plans, start=oos_start)

        # ---- statistics ------------------------------------------------------
        p_pool = np.concatenate(pooled_p)
        y_pool = np.concatenate(pooled_y)
        pooled_auc = float(roc_auc_score(y_pool, p_pool))
        pooled_brier = float(np.mean((p_pool - y_pool) ** 2))
        base_rate = float(y_pool.mean())
        gate = p_pool >= cfg.signals.min_probability
        high_conf_hit = float(y_pool[gate].mean()) if gate.any() else float("nan")

        boot = bootstrap_analysis(
            result.returns, n_sims=1000, avg_block=10.0, seed=seed, periods_per_year=ppy
        )

        returns = result.returns.dropna()
        sr_period = (
            float(returns.mean() / returns.std()) if returns.std() > 0 else 0.0
        )
        skew = float(sstats.skew(returns)) if len(returns) > 10 else 0.0
        kurt = float(sstats.kurtosis(returns, fisher=False)) if len(returns) > 10 else 3.0
        sr_var_proxy = float(np.var([r / np.sqrt(ppy) for r in [
            boot.sharpe_ci[0], boot.sharpe_median, boot.sharpe_ci[1]
        ]]))
        dsr_sensitivity = {
            f"n_trials={n}": deflated_sharpe_ratio(
                sr_period, sr_var_proxy, n, len(returns), skew, kurt
            )
            for n in (1, 5, 10, 25)
        }

        trades_per_year = (
            len(result.trades) / (len(result.equity) / ppy) if len(result.equity) else 0.0
        )
        ruin = risk_of_ruin(
            np.array([t.pnl_fraction for t in result.trades]),
            trades_per_year=max(trades_per_year, 1.0),
            seed=seed,
        )

        regime_breakdown = self._regime_breakdown(result.returns, regimes["regime"], ppy)

        if last_ensemble is not None and len(folds) > 0:
            f = folds[-1]
            importance = permutation_rank(
                last_ensemble,
                X_all.iloc[f.test_idx][last_selected],
                lab["label"].iloc[f.test_idx],
                n_repeats=3,
                seed=seed,
            )

        decisions = pd.DataFrame(decision_rows)
        report = WalkForwardReport(
            folds=fold_reports,
            pooled_auc=pooled_auc,
            pooled_brier=pooled_brier,
            high_conf_hit_rate=high_conf_hit,
            base_rate=base_rate,
            backtest=result,
            bootstrap=boot,
            dsr_sensitivity=dsr_sensitivity,
            ruin=ruin,
            regime_breakdown=regime_breakdown,
            signals=all_signals,
            importance=importance,
            regimes=regimes,
            decisions=decisions,
        )
        # Stash the final-fold artifacts so the scanner can reuse the most
        # recently trained model without refitting.
        self.last_fold_artifacts = {
            "ensemble": last_ensemble,
            "selected": last_selected,
            "explainer": last_explainer,
            "analogues": last_analogues,
        }
        return report

    # ------------------------------------------------------------------ #

    @staticmethod
    def _regime_breakdown(
        returns: pd.Series, regimes: pd.Series, periods_per_year: float = TRADING_DAYS
    ) -> dict[str, dict]:
        joined = pd.DataFrame({"ret": returns}).join(regimes.rename("regime"), how="left")
        joined["regime"] = joined["regime"].ffill()
        out: dict[str, dict] = {}
        for regime, group in joined.groupby("regime"):
            r = group["ret"].dropna()
            if len(r) < 5:
                continue
            ann = float(r.mean() * periods_per_year)
            sharpe = (
                float(r.mean() / r.std() * np.sqrt(periods_per_year)) if r.std() > 0 else 0.0
            )
            out[str(regime)] = {
                "days": len(r),
                "ann_return": round(ann, 4),
                "sharpe": round(sharpe, 3),
            }
        return out
