"""Margin mechanics: the arithmetic that decides whether an account survives.

Leverage is the one part of this platform where a sign error or an off-by-one
in a cap does not produce a slightly worse Sharpe — it produces a position that
gets closed by the exchange at a price nobody chose. So the invariants tested
here are deliberately blunt: risk scales with the multiple, liquidation always
sits beyond the stop, and no position can ever lose more than the cash behind
it.
"""

from __future__ import annotations

import pytest

from titan.core.config import LeverageConfig, TitanConfig, load_config
from titan.core.types import Side
from titan.risk.leverage import (
    LeverageTerms,
    liquidated,
    liquidation_distance,
    liquidation_price,
    plan,
    safe_leverage,
)

MMR = 0.005


# ------------------------------------------------------------ liquidation ----


def test_liquidation_distance_is_one_over_leverage_less_maintenance():
    assert liquidation_distance(3.0, MMR) == pytest.approx(1 / 3 - 0.005)
    assert liquidation_distance(10.0, MMR) == pytest.approx(0.095)


def test_a_long_liquidates_below_entry_and_a_short_above():
    long_liq = liquidation_price(100.0, 3.0, Side.LONG, MMR)
    short_liq = liquidation_price(100.0, 3.0, Side.SHORT, MMR)

    assert long_liq == pytest.approx(100.0 * (1 - (1 / 3 - 0.005)))
    assert short_liq == pytest.approx(100.0 * (1 + (1 / 3 - 0.005)))
    assert long_liq < 100.0 < short_liq


def test_spot_has_no_liquidation_price():
    """1x is cash. There is no level at which a broker closes it for you."""
    assert liquidation_price(100.0, 1.0, Side.LONG, MMR) is None


def test_higher_leverage_liquidates_sooner():
    d = [liquidation_distance(lev, MMR) for lev in (2.0, 3.0, 5.0, 10.0)]
    assert d == sorted(d, reverse=True)


# --------------------------------------------------------- safe leverage ----


def test_a_tight_stop_leaves_the_configured_ceiling_binding():
    """2% stop: liquidation at 3x is 33% away, nowhere near it."""
    assert safe_leverage(0.02, max_leverage=3.0, maintenance_margin_rate=MMR) == 3.0


def test_a_wide_stop_de_levers_itself():
    """A 25% stop cannot support 3x: liquidation would land inside it."""
    lev = safe_leverage(0.25, max_leverage=3.0, maintenance_margin_rate=MMR)
    assert lev < 3.0
    assert lev == pytest.approx(1 / (1.5 * 0.25 + 0.005))


def test_an_absurd_stop_falls_back_to_cash_rather_than_rejecting():
    assert safe_leverage(0.9, max_leverage=5.0, maintenance_margin_rate=MMR) == 1.0


def test_a_zero_stop_distance_cannot_be_levered():
    assert safe_leverage(0.0, max_leverage=5.0) == 1.0


@pytest.mark.parametrize("stop", [0.005, 0.01, 0.03, 0.08, 0.15, 0.30, 0.45])
@pytest.mark.parametrize("max_lev", [2.0, 3.0, 5.0, 10.0])
def test_liquidation_never_lands_inside_the_stop(stop, max_lev):
    """The invariant the whole design exists to protect.

    A position whose liquidation price is nearer than its stop does not have
    that stop — it has a liquidation dressed up as one, and the trade cannot
    lose the amount its order card claims.
    """
    terms = LeverageTerms(max_leverage=max_lev, maintenance_margin_rate=MMR, stop_buffer=1.5)
    p = plan(
        base_size=0.05, entry=100.0, stop_distance=stop, side=Side.LONG,
        holding_bars=10, terms=terms,
    )
    if not p.is_levered:
        assert p.liquidation_price is None
        return
    assert p.liquidation_distance >= 1.5 * stop - 1e-12
    assert p.liquidation_price < 100.0 * (1 - stop)


# ----------------------------------------------------------------- plan ----


def test_leverage_multiplies_notional_and_leaves_margin_at_the_base_size():
    terms = LeverageTerms(max_leverage=3.0, maintenance_margin_rate=MMR)
    p = plan(
        base_size=0.05, entry=100.0, stop_distance=0.02, side=Side.LONG,
        holding_bars=10, terms=terms,
    )
    assert p.leverage == 3.0
    assert p.notional_fraction == pytest.approx(0.15)
    assert p.margin_fraction == pytest.approx(0.05)


def test_leverage_multiplies_risk_too():
    """The honesty check. 3x notional on the same stop is 3x the loss.

    If this ever passes with equal risk at both multiples, the order card is
    understating what the trade can cost and every downstream risk number is
    fiction.
    """
    spot = plan(
        base_size=0.05, entry=100.0, stop_distance=0.02, side=Side.LONG,
        holding_bars=10, terms=LeverageTerms.spot(),
    )
    levered = plan(
        base_size=0.05, entry=100.0, stop_distance=0.02, side=Side.LONG,
        holding_bars=10, terms=LeverageTerms(max_leverage=3.0, maintenance_margin_rate=MMR),
    )
    assert spot.risk_fraction_of_equity == pytest.approx(0.001)
    assert levered.risk_fraction_of_equity == pytest.approx(0.003)
    assert levered.risk_fraction_of_equity == pytest.approx(3 * spot.risk_fraction_of_equity)


def test_spot_terms_pass_the_size_through_untouched():
    p = plan(
        base_size=0.08, entry=50.0, stop_distance=0.04, side=Side.LONG,
        holding_bars=10, terms=LeverageTerms.spot(),
    )
    assert p.leverage == 1.0
    assert p.notional_fraction == p.margin_fraction == 0.08
    assert p.liquidation_price is None
    assert p.funding_cost == 0.0


def test_funding_accrues_with_the_holding_period():
    terms = LeverageTerms(max_leverage=3.0, funding_per_bar=0.0003, maintenance_margin_rate=MMR)
    short_hold = plan(base_size=0.05, entry=100.0, stop_distance=0.02,
                      side=Side.LONG, holding_bars=1, terms=terms)
    long_hold = plan(base_size=0.05, entry=100.0, stop_distance=0.02,
                     side=Side.LONG, holding_bars=30, terms=terms)

    assert short_hold.funding_cost == pytest.approx(0.0003)
    assert long_hold.funding_cost == pytest.approx(0.009)


# -------------------------------------------------------------- config ----


def test_terms_resolve_per_asset_class():
    cfg = LeverageConfig(max_leverage={"crypto": 3.0})
    assert LeverageTerms.resolve(cfg, "crypto", "1d").max_leverage == 3.0
    assert LeverageTerms.resolve(cfg, "equity", "1d").max_leverage == 1.0
    assert not LeverageTerms.resolve(cfg, "equity", "1d").enabled


def test_funding_converts_through_wall_clock_not_the_calendar():
    """A 3h bar rents the notional for 3h — an eighth of the daily rate."""
    cfg = LeverageConfig(max_leverage={"crypto": 3.0}, funding_bps_daily=3.0)
    daily = LeverageTerms.resolve(cfg, "crypto", "1d")
    three_hour = LeverageTerms.resolve(cfg, "crypto", "3h")

    assert daily.funding_per_bar == pytest.approx(0.0003)
    assert three_hour.funding_per_bar == pytest.approx(0.0003 / 8)


def test_an_empty_leverage_config_means_cash_everywhere():
    cfg = LeverageConfig()
    for asset_class in ("crypto", "equity", "etf", "future"):
        assert not LeverageTerms.resolve(cfg, asset_class, "1d").enabled


@pytest.mark.parametrize("bad", [0.5, 0.0, 25.0, 100.0])
def test_config_rejects_leverage_outside_the_sane_band(bad):
    with pytest.raises(ValueError, match=r"between 1\.0"):
        LeverageConfig(max_leverage={"crypto": bad})


def test_default_config_is_unlevered():
    """Margin must be opted into. A default config trades cash."""
    cfg = TitanConfig()
    assert cfg.risk.leverage.max_leverage == {}
    assert cfg.risk.leverage.for_asset_class("crypto") == 1.0


def test_the_shipped_top100_config_levers_both_books():
    """Crypto perps carry more than Reg T equity margin, and both are on."""
    cfg = load_config("configs/top100.yaml")
    equity = cfg.risk.leverage.for_asset_class("equity")
    crypto = cfg.risk.leverage.for_asset_class("crypto")

    assert equity > 1.0 and crypto > equity
    assert equity <= 2.0, "above Reg T no retail equity account can post this"
    assert cfg.scanner.quota_for("equity") == cfg.scanner.quota_for("crypto")


# ------------------------------------------- an aggressive posture, checked ---
#
# Raising risk is a config edit; raising it COHERENTLY is not. Every dial below
# has a partner that will silently swallow it if left behind — a heat cap that
# truncates the sizes, a cash ceiling that cannot fund the notional ceiling, a
# gate loosened past the point where the trade is still positive-expectancy.
# None of those announce themselves: the account just quietly does less than
# the config says, which is the worst way to run a risk setting.


def _shipped():
    return load_config("configs/top100.yaml")


def test_the_heat_cap_does_not_silently_truncate_the_configured_sizes():
    """The failure mode: sizes raised, aggregate cap left behind.

    portfolio_heat_cap_pct bounds the SUM of size x stop-distance. If a full
    book's heat exceeds it, the risk engine scales positions down and nothing
    reports that the sizes in the config were never the sizes that traded.
    """
    import numpy as np

    from titan.risk.sizing import atr_risk_size, fractional_kelly, vol_target_size

    cfg = _shipped()
    r, payoff = cfg.risk, cfg.labels.tp_sigma / cfg.labels.sl_sigma

    worst = 0.0
    for sigma in (0.008, 0.012, 0.018, 0.025, 0.035, 0.050):
        stop = cfg.labels.sl_sigma * sigma
        base = min(
            fractional_kelly(0.65, payoff, r.kelly_fraction, r.max_position_weight),
            vol_target_size(sigma * np.sqrt(252.0), r.target_annual_vol,
                            cfg.backtest.max_positions, r.max_position_weight),
            atr_risk_size(stop, r.risk_per_trade_pct, r.max_position_weight),
        )
        worst = max(worst, base * stop)

    full_book = 100.0 * worst * cfg.backtest.max_positions
    assert full_book <= r.portfolio_heat_cap_pct, (
        f"a full book carries {full_book:.1f}% heat against a "
        f"{r.portfolio_heat_cap_pct}% cap: the risk engine will scale positions "
        "down and the configured sizes are fiction"
    )


def test_the_cash_ceiling_can_fund_the_notional_ceiling():
    """Two caps, and the wrong one binding means orders are refused on entry.

    max_account_leverage bounds notional; max_gross_exposure bounds the cash
    posted behind it. Margin is notional/L, so the LEAST levered book is the
    expensive one — an all-equity book at 2x needs half its notional in cash.
    """
    cfg = _shipped()
    lev = cfg.risk.leverage
    least_levered = min(lev.max_leverage.values())
    cash_needed = lev.max_account_leverage / least_levered

    assert cfg.backtest.max_gross_exposure >= cash_needed, (
        f"funding {lev.max_account_leverage}x notional at {least_levered}x needs "
        f"{cash_needed:.2f}x cash but max_gross_exposure is "
        f"{cfg.backtest.max_gross_exposure}x: the account refuses orders the "
        "scanner was told to emit"
    )

    # The consistency check above is satisfiable from either end, and raising
    # the cash cap is the cheaper-looking way to satisfy it. It is also the
    # wrong one: leverage is already modelled on notional, so margin above 1x
    # equity applies the multiple a second time and posts collateral the
    # balance does not hold. Pin that end down, or the pair can be made
    # consistent and unreachable at the same time.
    assert cfg.backtest.max_gross_exposure <= 1.0, (
        f"max_gross_exposure is {cfg.backtest.max_gross_exposure}x: the account "
        "would post more cash as margin than the balance holds, and every equity "
        "curve drawn from it is unreachable with the stated starting capital"
    )


def test_liquidation_still_sits_beyond_the_stop_at_the_raised_multiples():
    """The invariant leverage exists to not break, re-checked at 4x."""
    cfg = _shipped()
    lev = cfg.risk.leverage

    for asset_class, ceiling in lev.max_leverage.items():
        for stop in (0.01, 0.02, 0.05, 0.10, 0.20, 0.35):
            resolved = safe_leverage(
                stop, max_leverage=ceiling,
                maintenance_margin_rate=lev.maintenance_margin_rate,
                stop_buffer=lev.stop_buffer,
            )
            d_liq = liquidation_distance(resolved, lev.maintenance_margin_rate)
            assert d_liq > stop, (
                f"{asset_class} at {resolved:.2f}x liquidates at {d_liq:.1%}, "
                f"inside a {stop:.1%} stop — the stop could never fill"
            )


def test_the_gate_stays_above_break_even_after_being_loosened():
    """Riskier must still mean positive-expectancy, not a coin flip.

    Break-even under the barrier geometry is p = b/(a+b) before costs. A
    min_probability at or under that admits trades with no edge at all, which
    is not a risk/return trade — it is just losing money faster.
    """
    cfg = _shipped()
    a = cfg.labels.tp_sigma
    b = cfg.labels.sl_sigma
    break_even = b / (a + b)

    assert cfg.signals.min_probability > break_even + 0.05, (
        f"min_probability {cfg.signals.min_probability} is not clear of the "
        f"{break_even:.3f} break-even for tp={a}/sl={b}"
    )


def test_the_two_guards_that_do_not_move_with_risk_appetite():
    """A crash multiplier of zero and the Venn-ABERS lower bound.

    Neither trades risk for return. The crash state is where the model's
    calibration is known to be worthless, and the conservative gate is what
    separates a real edge from one that only exists if thin calibration data is
    taken on faith. Raising risk is a choice; deleting these is a different one.
    """
    cfg = _shipped()

    assert cfg.risk.regime_multipliers["crash"] == 0.0
    assert cfg.signals.conservative_gate is True
    assert cfg.risk.leverage.stop_buffer >= 1.25


# ---------------------------------------------------------- liquidated ----


def test_liquidated_fires_only_past_the_margin():
    assert not liquidated(-0.20, 3.0, MMR)
    assert liquidated(-0.35, 3.0, MMR)


def test_spot_is_never_liquidated():
    assert not liquidated(-0.99, 1.0, MMR)
