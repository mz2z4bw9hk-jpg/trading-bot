"""Typed configuration for the whole platform.

Every component receives its own config object (dependency injection); nothing
reads global state. Configs are loaded from YAML and validated by pydantic, so
a malformed research config fails loudly at startup instead of silently
producing garbage research.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from titan.core.timeframe import (
    UNVALIDATED_TIMEFRAMES,
    BarClock,
    Timeframe,
    is_finer,
    resolve_bars_per_year,
)


class RunConfig(BaseModel):
    seed: int = 7
    artifacts_dir: Path = Path("artifacts")
    log_level: str = "INFO"


class DataConfig(BaseModel):
    provider: Literal["synthetic", "yahoo", "csv"] = "synthetic"
    cache_dir: Path = Path("data_cache")
    bars: int = 2000
    min_history_bars: int = 420
    min_reliability: float = Field(0.70, ge=0.0, le=1.0)
    max_forward_fill: int = 2
    csv_dir: Path | None = None
    # What one bar means in wall-clock time. Everything annualized derives
    # from this; it is part of the research fingerprint, so a model trained on
    # 1h bars will refuse to scan a 1d config.
    timeframe: Timeframe = "1d"
    # Bars per year. None resolves from the timeframe and universe composition
    # (a 24/7 crypto book annualizes differently from a 6.5h equity session).
    bars_per_year: float | None = Field(None, gt=0)
    # Required to run 1m/5m research: those intervals break the platform's
    # fill and cost assumptions rather than merely straining them.
    acknowledge_unvalidated_timeframe: bool = False
    # Fetch at this (finer) interval and aggregate up to `timeframe`. Two uses:
    # reaching a bar the vendor does not serve (Yahoo has no 2h/3h/4h), and
    # repairing a sparsely-populated volume column by summing source bars.
    resample_from: Timeframe | None = None
    # Synthetic provider only: innovation scale of the planted AR(1) drift.
    # None uses the generator default (realistically weak). Larger values are
    # for control experiments: the pipeline MUST detect strong planted signal.
    synthetic_drift_sigma: float | None = Field(None, gt=0)

    @model_validator(mode="after")
    def _check_resample(self) -> DataConfig:
        if self.resample_from is None:
            return self
        if not is_finer(self.resample_from, self.timeframe):
            raise ValueError(
                f"data.resample_from ({self.resample_from!r}) must be a FINER interval "
                f"than data.timeframe ({self.timeframe!r}); bars can only be aggregated, "
                "never subdivided"
            )
        return self

    @model_validator(mode="after")
    def _check_timeframe(self) -> DataConfig:
        if self.timeframe in UNVALIDATED_TIMEFRAMES and not self.acknowledge_unvalidated_timeframe:
            raise ValueError(
                f"timeframe {self.timeframe!r} is not validated research on this platform: "
                "fills are modelled at the next bar's open (fiction at this frequency), "
                "the cost model is calibrated for daily turnover (the spread would consume "
                "the entire edge), and OHLCV bars omit the order-book state that drives "
                "sub-minute price formation. Set data.acknowledge_unvalidated_timeframe: "
                "true to run it anyway — the numbers are exploratory, not evidence."
            )
        return self


class UniverseItem(BaseModel):
    symbol: str
    asset_class: str = "equity"
    sector: str = "unknown"


class UniverseConfig(BaseModel):
    name: str = "default"
    benchmark: str = "INDEX"
    instruments: list[UniverseItem] = Field(default_factory=list)


class FeatureConfig(BaseModel):
    momentum_windows: list[int] = Field(default_factory=lambda: [5, 10, 21, 63, 126, 252])
    trend_windows: list[int] = Field(default_factory=lambda: [10, 21, 50, 100])
    meanrev_windows: list[int] = Field(default_factory=lambda: [5, 10, 21])
    vol_windows: list[int] = Field(default_factory=lambda: [5, 10, 21, 63])
    volume_windows: list[int] = Field(default_factory=lambda: [10, 21, 63])
    structure_windows: list[int] = Field(default_factory=lambda: [21, 63, 126])
    cross_windows: list[int] = Field(default_factory=lambda: [21, 63, 126])
    redundancy_threshold: float = Field(0.90, ge=0.5, le=1.0)
    max_features: int = 48
    importance_method: Literal["permutation", "model"] = "permutation"
    min_periods_fraction: float = Field(0.8, gt=0.0, le=1.0)


class LabelConfig(BaseModel):
    horizon_bars: int = 15
    tp_sigma: float = Field(2.0, gt=0)
    sl_sigma: float = Field(1.5, gt=0)
    vol_span: int = 21
    min_vol_floor: float = 1e-4


class CVConfig(BaseModel):
    scheme: Literal["expanding", "rolling"] = "expanding"
    n_folds: int = Field(5, ge=2)
    embargo_bars: int = 5
    min_train_bars: int = 252
    test_bars: int = 126


MemberName = Literal["hgb", "rf", "logistic"]


def _default_members() -> list[MemberName]:
    return ["hgb", "rf", "logistic"]


class ModelConfig(BaseModel):
    members: list[MemberName] = Field(default_factory=_default_members)
    calibration: Literal["isotonic", "sigmoid"] = "isotonic"
    tuning_iterations: int = Field(10, ge=0)
    tuning_metric: Literal["log_loss", "auc"] = "log_loss"
    internal_folds: int = Field(3, ge=2)  # purged OOF folds inside each train window
    store_dir: Path = Path("models_store")
    max_train_rows: int = 250_000


class RegimeConfig(BaseModel):
    n_states: int = Field(4, ge=2, le=8)
    trend_window: int = 63
    vol_window: int = 21
    smoothing_halflife: float = Field(3.0, gt=0)
    min_train_bars: int = 252
    crash_vol_percentile: float = Field(0.95, gt=0.5, lt=1.0)
    crash_drawdown: float = Field(-0.15, lt=0)
    correction_drawdown: float = Field(-0.08, lt=0)


class CostConfig(BaseModel):
    commission_bps: float = Field(1.0, ge=0)
    spread_bps: float = Field(2.5, ge=0)
    impact_coefficient: float = Field(0.10, ge=0)
    borrow_bps_daily: float = Field(0.5, ge=0)


class BacktestConfig(BaseModel):
    initial_capital: float = Field(1_000_000.0, gt=0)
    costs: CostConfig = Field(default_factory=CostConfig)
    max_positions: int = Field(10, ge=1)
    max_gross_exposure: float = Field(1.0, gt=0)
    allow_short: bool = False
    execution_lag_bars: int = Field(1, ge=1)  # decide at close t, execute at open t+lag
    stop_first_on_ambiguous_bar: bool = True  # pessimistic intrabar assumption


class RiskConfig(BaseModel):
    risk_per_trade_pct: float = Field(0.5, gt=0, le=5.0)  # percent of equity at stop
    kelly_fraction: float = Field(0.25, gt=0, le=1.0)
    target_annual_vol: float = Field(0.12, gt=0)
    max_position_weight: float = Field(0.15, gt=0, le=1.0)
    portfolio_heat_cap_pct: float = Field(4.0, gt=0)  # sum of open risk, % of equity
    correlation_penalty_threshold: float = Field(0.60, ge=0, le=1.0)
    max_sector_weight: float = Field(0.35, gt=0, le=1.0)
    dd_throttle_start: float = Field(0.05, gt=0)
    dd_throttle_full: float = Field(0.15, gt=0)
    var_confidence: float = Field(0.95, gt=0.5, lt=1.0)
    regime_multipliers: dict[str, float] = Field(
        default_factory=lambda: {
            "strong_bull": 1.00,
            "bull": 1.00,
            "weak_bull": 0.80,
            "accumulation": 0.80,
            "range": 0.60,
            "distribution": 0.45,
            "correction": 0.40,
            "bear": 0.30,
            "crash": 0.0,
        }
    )


class TechnicalConfig(BaseModel):
    """Rule-based swing setups: a second order source, off by default.

    These bypass the calibrated EV gate by construction — they are a different
    hypothesis, not a better-tuned version of the same one — so they are opt-in
    and every order they produce is labelled with the rule that fired.
    """

    enabled: bool = False
    setups: list[str] = Field(
        default_factory=lambda: [
            "donchian_breakout",
            "pullback_in_uptrend",
            "ma_cross",
            "oversold_bounce",
            "macd_momentum",
        ]
    )
    # Cap on technical orders per scan, taken strongest-first. Without a cap a
    # 178-name universe can fire dozens on a trending day and the account would
    # be fully committed to one day's worth of setups.
    max_orders_per_scan: int = Field(5, ge=1)
    # A rule that pays less at its second target than it risks at its stop is
    # not a trade, however cleanly the pattern printed.
    min_risk_reward: float = Field(1.5, gt=0)
    # Rules do not read the regime detector, but the platform still refuses to
    # buy breakouts into a crash.
    skip_in_crash: bool = True


class SignalConfig(BaseModel):
    min_probability: float = Field(0.55, gt=0.5, lt=1.0)
    ev_margin_bps: float = Field(5.0, ge=0)
    analogue_k: int = Field(50, ge=10)
    max_uncertainty: float = Field(0.25, gt=0)
    # Require the LOWER Venn-ABERS probability bound to clear the adaptive
    # gate, not just the point estimate: a signal that only exists if thin
    # calibration data is taken on faith does not deserve to exist.
    conservative_gate: bool = True
    grade_thresholds: dict[str, float] = Field(
        default_factory=lambda: {"A+": 85.0, "A": 75.0, "B+": 65.0, "B": 55.0}
    )
    technical: TechnicalConfig = Field(default_factory=TechnicalConfig)


class ScannerConfig(BaseModel):
    top_n: int = Field(10, ge=1)
    # How many panel dates an instrument's own latest bar may lag the newest
    # date in the panel before the scanner reports it as stale instead of
    # ranking it. Mixed-calendar universes need headroom: crypto trades every
    # day, so after a long weekend an equity's freshest bar is 3-4 panel dates
    # old and is still perfectly current for that instrument.
    max_staleness_bars: int = Field(5, ge=0)


class MonitorConfig(BaseModel):
    psi_alert: float = Field(0.25, gt=0)
    psi_warn: float = Field(0.10, gt=0)
    min_live_samples: int = Field(50, ge=10)
    promotion_p_value: float = Field(0.05, gt=0, lt=0.5)
    # Starting balance of the forward paper account replayed from the tracking
    # log. Separate from backtest.initial_capital: that funds a historical
    # simulation, this funds the live-forward one.
    paper_starting_equity: float = Field(1_000_000.0, gt=0)


class TitanConfig(BaseModel):
    """Root configuration object injected throughout the platform."""

    run: RunConfig = Field(default_factory=RunConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    universe: UniverseConfig = Field(default_factory=UniverseConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    labels: LabelConfig = Field(default_factory=LabelConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    regime: RegimeConfig = Field(default_factory=RegimeConfig)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    signals: SignalConfig = Field(default_factory=SignalConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)

    @field_validator("labels")
    @classmethod
    def _label_barriers_sane(cls, v: LabelConfig) -> LabelConfig:
        if v.horizon_bars < 2:
            raise ValueError("label horizon must be >= 2 bars")
        return v


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def bar_clock(cfg: TitanConfig) -> BarClock:
    """Resolve the bar interval and its annualization constant.

    A universe whose tradeable instruments are all crypto is annualized on a
    24/7 calendar; anything with an equity, ETF or futures leg uses the
    session calendar. The benchmark is excluded from that test — crypto books
    are routinely benchmarked against SPY, and that alone should not flip the
    convention for the whole run.
    """
    tradeable = [i for i in cfg.universe.instruments if i.symbol != cfg.universe.benchmark]
    continuous = bool(tradeable) and all(i.asset_class == "crypto" for i in tradeable)
    return BarClock(
        timeframe=cfg.data.timeframe,
        bars_per_year=resolve_bars_per_year(
            cfg.data.timeframe, continuous=continuous, override=cfg.data.bars_per_year
        ),
        continuous=continuous,
    )


def research_fingerprint(cfg: TitanConfig) -> str:
    """Fingerprint of the *world* a model is trained for.

    Covers the sections that define what the data and labels mean — data
    source, universe, label geometry, feature settings, and (synthetic only,
    where the seed literally generates the market) the seed. Two configs with
    the same fingerprint produce models and scans that may be compared;
    scanning a model against a config with a different fingerprint produces
    numbers about a world the model never saw. Deliberately excludes run
    paths, CV/model/tuning, risk and gate settings: those change how hard we
    look, not what we are looking at.
    """
    payload: dict[str, Any] = {
        "data": {k: v for k, v in cfg.data.model_dump(mode="json").items() if k != "cache_dir"},
        "universe": cfg.universe.model_dump(mode="json"),
        "labels": cfg.labels.model_dump(mode="json"),
        "features": cfg.features.model_dump(mode="json"),
    }
    if cfg.data.provider == "synthetic":
        payload["seed"] = cfg.run.seed
    blob = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> TitanConfig:
    """Load configuration from YAML, applying optional dict overrides on top."""
    raw: dict[str, Any] = {}
    if path is not None:
        text = Path(path).read_text()
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping, got {type(loaded)}")
        raw = loaded
    if overrides:
        raw = _deep_merge(raw, overrides)
    return TitanConfig.model_validate(raw)
