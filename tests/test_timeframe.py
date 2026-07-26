"""Bar interval: annualization, universe convention, and the scan guard.

The failure this guards against is silent. A hard-coded 252 on hourly bars
does not raise; it reports a Sharpe inflated by sqrt(6.5) and vol-targeted
positions oversized by the same factor, and every downstream number looks
perfectly reasonable.
"""

from __future__ import annotations

import copy
import pathlib

import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from titan.backtest.metrics import summarize
from titan.core.config import bar_clock, load_config, research_fingerprint
from titan.core.timeframe import (
    BarClock,
    empirical_bars_per_year,
    resolve_bars_per_year,
    session_bars_per_day,
    warn_on_calendar_mismatch,
)


def _write(tmp_path, cfg_dict, name="cfg.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(cfg_dict))
    return path


@pytest.fixture
def base_cfg():
    return yaml.safe_load(open("configs/default.yaml"))


# ---------------------------------------------------------------- tables --


def test_daily_session_annualization_is_the_trading_year():
    assert resolve_bars_per_year("1d", continuous=False) == 252.0


def test_continuous_market_annualizes_on_the_calendar_year():
    assert resolve_bars_per_year("1d", continuous=True) == 365.0


def test_hourly_differs_from_daily_by_the_session_length():
    hourly = resolve_bars_per_year("1h", continuous=False)
    assert hourly == pytest.approx(252 * 6.5)


def test_override_wins_over_the_table():
    assert resolve_bars_per_year("1h", continuous=False, override=1000.0) == 1000.0


def test_unknown_timeframe_is_rejected():
    with pytest.raises(ValueError, match="unknown timeframe"):
        resolve_bars_per_year("7h", continuous=False)


def test_multi_hour_timeframes_resolve():
    """2h/3h/4h exist only via aggregation, but must annualize correctly."""
    assert resolve_bars_per_year("2h", continuous=True) == pytest.approx(365 * 12)
    assert resolve_bars_per_year("2h", continuous=False) == pytest.approx(252 * 3.25)


def test_session_bars_per_day():
    assert session_bars_per_day("1d") == pytest.approx(1.0)
    assert session_bars_per_day("1h") == pytest.approx(6.5)


# ------------------------------------------------------- clock resolution --


def test_default_config_is_unchanged_by_the_timeframe_work(base_cfg, tmp_path):
    """The daily path must still annualize at exactly 252."""
    clock = bar_clock(load_config(_write(tmp_path, base_cfg)))
    assert clock.timeframe == "1d"
    assert clock.bars_per_year == 252.0
    assert clock.continuous is False


def test_all_crypto_universe_selects_the_continuous_calendar(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    cfg["universe"]["benchmark"] = "SPY"
    cfg["universe"]["instruments"] = [
        {"symbol": "SPY", "asset_class": "etf", "sector": "broad"},
        {"symbol": "BTC-USD", "asset_class": "crypto", "sector": "crypto"},
        {"symbol": "ETH-USD", "asset_class": "crypto", "sector": "crypto"},
    ]
    # The benchmark is an ETF but is excluded from the test: crypto books are
    # routinely benchmarked against SPY and that must not flip the convention.
    assert bar_clock(load_config(_write(tmp_path, cfg))).continuous is True


def test_one_equity_leg_reverts_to_the_session_calendar(base_cfg, tmp_path):
    cfg = copy.deepcopy(base_cfg)
    cfg["universe"]["benchmark"] = "SPY"
    cfg["universe"]["instruments"] = [
        {"symbol": "SPY", "asset_class": "etf", "sector": "broad"},
        {"symbol": "BTC-USD", "asset_class": "crypto", "sector": "crypto"},
        {"symbol": "MU", "asset_class": "equity", "sector": "technology"},
    ]
    assert bar_clock(load_config(_write(tmp_path, cfg))).continuous is False


# ------------------------------------------------------------- refusal ----


@pytest.mark.parametrize("timeframe", ["1m", "5m"])
def test_scalping_timeframes_are_refused_by_default(base_cfg, tmp_path, timeframe):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["timeframe"] = timeframe
    with pytest.raises(ValidationError, match="not validated research"):
        load_config(_write(tmp_path, cfg))


@pytest.mark.parametrize("timeframe", ["1m", "5m"])
def test_scalping_runs_once_explicitly_acknowledged(base_cfg, tmp_path, timeframe):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["timeframe"] = timeframe
    cfg["data"]["acknowledge_unvalidated_timeframe"] = True
    assert load_config(_write(tmp_path, cfg)).data.timeframe == timeframe


@pytest.mark.parametrize("timeframe", ["1wk", "1d", "4h", "1h", "30m", "15m"])
def test_supported_timeframes_need_no_acknowledgement(base_cfg, tmp_path, timeframe):
    cfg = copy.deepcopy(base_cfg)
    cfg["data"]["timeframe"] = timeframe
    assert load_config(_write(tmp_path, cfg)).data.timeframe == timeframe


# --------------------------------------------------------- the scan guard --


def test_timeframe_is_part_of_the_research_fingerprint(base_cfg, tmp_path):
    """A model trained on 1h must refuse to scan a 1d config."""
    daily = load_config(_write(tmp_path, base_cfg, "a.yaml"))
    hourly_cfg = copy.deepcopy(base_cfg)
    hourly_cfg["data"]["timeframe"] = "1h"
    hourly = load_config(_write(tmp_path, hourly_cfg, "b.yaml"))

    assert research_fingerprint(daily) != research_fingerprint(hourly)


def test_bars_per_year_override_is_part_of_the_fingerprint(base_cfg, tmp_path):
    """It rescales every annualized number, so it defines the world too."""
    plain = load_config(_write(tmp_path, base_cfg, "a.yaml"))
    tweaked_cfg = copy.deepcopy(base_cfg)
    tweaked_cfg["data"]["bars_per_year"] = 365.0
    tweaked = load_config(_write(tmp_path, tweaked_cfg, "b.yaml"))

    assert research_fingerprint(plain) != research_fingerprint(tweaked)


# ------------------------------------------------------ empirical checks --


def test_empirical_bars_per_year_counts_bars_per_calendar_span():
    """Business-day bars must imply ~252/year, not 365 — weekends are gaps."""
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=252 * 3))
    assert empirical_bars_per_year(idx) == pytest.approx(252, rel=0.05)


def test_empirical_bars_per_year_handles_a_continuous_calendar():
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=365 * 2, freq="D"))
    assert empirical_bars_per_year(idx) == pytest.approx(365, rel=0.05)


def test_empirical_bars_per_year_declines_to_guess_from_a_short_series():
    assert empirical_bars_per_year(pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=5))) is None


def test_calendar_mismatch_is_logged(caplog):
    """A 24/7 book annualized on a session calendar must not pass silently."""
    clock = BarClock(timeframe="1d", bars_per_year=252.0, continuous=False)
    idx = pd.DatetimeIndex(pd.date_range("2020-01-01", periods=365 * 3, freq="D"))
    with caplog.at_level("WARNING"):
        warn_on_calendar_mismatch(clock, idx)
    assert "calendar mismatch" in caplog.text


def test_matching_calendar_is_quiet(caplog):
    clock = BarClock(timeframe="1d", bars_per_year=252.0, continuous=False)
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=252 * 3))
    with caplog.at_level("WARNING"):
        warn_on_calendar_mismatch(clock, idx)
    assert "calendar mismatch" not in caplog.text


# ------------------------------------------------------------ the payoff --


def test_annualization_actually_changes_the_reported_sharpe():
    """The whole point: identical returns, different bar, different Sharpe."""
    rng = np.random.default_rng(7)
    equity = pd.Series(1e6 * np.exp(np.cumsum(rng.normal(2e-4, 4e-3, 800))))

    daily = summarize(equity, trades=[], periods_per_year=252.0)
    hourly = summarize(equity, trades=[], periods_per_year=252.0 * 6.5)

    assert hourly.sharpe == pytest.approx(daily.sharpe * np.sqrt(6.5), rel=1e-6)
    assert hourly.ann_vol == pytest.approx(daily.ann_vol * np.sqrt(6.5), rel=1e-6)


# ---------------------------------------------------- yahoo vendor limits --


def test_yahoo_rejects_a_timeframe_it_has_no_native_bar_for():
    from titan.data.providers import YahooProvider

    with pytest.raises(ValueError, match="no native 4h bar"):
        YahooProvider("4h")


def test_yahoo_clamps_intraday_requests_to_the_vendor_cap(caplog):
    """Asking for more intraday history than Yahoo keeps returns less, silently."""
    from titan.data.providers import YahooProvider

    provider = YahooProvider("1h")
    with caplog.at_level("WARNING"):
        days = provider._period_days(10_000)  # ~6 years of hourly bars

    assert days == 730  # Yahoo's hourly cap
    assert "at most 730 days" in caplog.text


def test_yahoo_leaves_a_request_inside_the_cap_alone(caplog):
    from titan.data.providers import YahooProvider

    provider = YahooProvider("1h")
    with caplog.at_level("WARNING"):
        days = provider._period_days(2500)  # ~1 year of hourly bars

    assert days < 730
    assert "at most" not in caplog.text


def test_yahoo_does_not_clamp_daily_requests(caplog):
    from titan.data.providers import YahooProvider

    provider = YahooProvider("1d")
    with caplog.at_level("WARNING"):
        days = provider._period_days(2500)

    assert days > 2500  # fetched generously, then trimmed
    assert "at most" not in caplog.text


# ------------------------------------------------- shipped style configs --

STYLE_CONFIGS = sorted(pathlib.Path("configs").glob("style-*.yaml"))

# The validated daily config pairs tp 2.0 / sl 1.5 with a 10-bar horizon.
_DAILY_TP_REACH = 2.0 / np.sqrt(10)
_DAILY_SL_REACH = 1.5 / np.sqrt(10)


def test_style_configs_exist():
    assert STYLE_CONFIGS, "expected shipped style-*.yaml configs"


@pytest.mark.parametrize("path", STYLE_CONFIGS, ids=lambda p: p.stem)
def test_style_config_loads_and_resolves_a_clock(path):
    assert bar_clock(load_config(path)).bars_per_year > 0


@pytest.mark.parametrize("path", STYLE_CONFIGS, ids=lambda p: p.stem)
def test_style_config_fold_geometry_fits_its_history_floor(path):
    """A config whose folds need more bars than QC guarantees cannot run."""
    cfg = load_config(path)
    needed = cfg.cv.min_train_bars + cfg.cv.n_folds * cfg.cv.test_bars
    assert needed <= cfg.data.min_history_bars, (
        f"{path.name}: folds need {needed} bars but min_history_bars is "
        f"{cfg.data.min_history_bars}"
    )


@pytest.mark.parametrize("path", STYLE_CONFIGS, ids=lambda p: p.stem)
def test_style_config_barriers_stay_reachable_within_the_horizon(path):
    """Barriers are per-bar sigma and are NOT horizon-scaled.

    Only sqrt(horizon) sigmas of cumulative move are available, so carrying
    the daily tp_sigma onto a short horizon asks price to travel further than
    the horizon allows: nearly every label times out and the model trains on
    an almost-constant target. Each config must hold roughly the same
    reachability ratio as the validated daily geometry.
    """
    cfg = load_config(path)
    horizon = cfg.labels.horizon_bars
    tp_reach = cfg.labels.tp_sigma / np.sqrt(horizon)
    sl_reach = cfg.labels.sl_sigma / np.sqrt(horizon)

    assert tp_reach == pytest.approx(_DAILY_TP_REACH, rel=0.35), (
        f"{path.name}: tp_sigma {cfg.labels.tp_sigma} over a {horizon}-bar horizon "
        f"gives reachability {tp_reach:.3f} vs the validated {_DAILY_TP_REACH:.3f}"
    )
    assert sl_reach == pytest.approx(_DAILY_SL_REACH, rel=0.35)
