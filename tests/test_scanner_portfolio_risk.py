"""The live scanner must size its orders as a portfolio, not as N lone bets.

``risk/portfolio.py`` opens by claiming "the same object serves live scanning
and backtesting — one code path, no sim-vs-prod drift." That was false. The
RiskEngine was constructed in exactly one place, ``backtest/walkforward.py``,
and the scanner emitted the signal generator's standalone sizes untouched.

The generator sizes each candidate as the minimum of Kelly, a vol target and
an ATR stop budget. All three are properties of one instrument. Nothing in
that set can see the rest of the book, so nothing in it can notice that five
candidates are the same trade. The observed symptom was a live paper account
holding CVX, PG, LLY, CB and MRK simultaneously — five large-cap defensives,
individually within every per-position limit, collectively one leveraged bet
on one factor, and all five red on the same day.

These tests pin the live path to the portfolio-level controls: correlation,
heat, sector, and regime appetite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from titan.core.config import TitanConfig
from titan.core.types import Regime, Side, TradeGrade, VolState
from titan.features.registry import FeaturePanel
from titan.signals.schema import Signal

DATES = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=120), tz="UTC")


class _StubEnsemble:
    has_intervals = False

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 0.35), np.full(len(X), 0.65)])

    def uncertainty(self, X):
        return np.full(len(X), 0.01)


class _StubDetector:
    def __init__(self, regime: Regime = Regime.BULL) -> None:
        self._regime = regime

    def snapshot(self, frame):
        regime = self._regime

        class _S:
            pass

        _S.regime = regime
        _S.vol_state = VolState.NORMAL
        _S.confidence = 0.9
        return _S()


class _FixedSizeGenerator:
    """Emits one identical, well-formed order per symbol.

    Identical on purpose: if every candidate asks for the same size, any
    difference in the emitted sizes is the portfolio layer, which is the only
    thing under test here.
    """

    def __init__(self, size: float = 0.20, leverage: float = 2.0) -> None:
        self._size = size
        self._leverage = leverage

    def generate(self, *, symbol, date, asset_class, **kwargs):
        entry = 100.0
        stop = 96.0                       # 4% stop -> 0.8% risk at 0.20 notional
        return Signal(
            symbol=symbol,
            date=date,
            side=Side.LONG,
            asset_class=asset_class,
            confidence_score=70.0,
            trade_grade=TradeGrade.A,
            market_entry=entry,
            stop_loss=stop,
            take_profit_levels=[entry * 1.06],
            position_size_fraction=self._size,
            leverage=self._leverage,
            margin_fraction=self._size / self._leverage,
            risk_percentage=100.0 * self._size * (entry - stop) / entry,
            liquidation_price=80.0,
            expected_holding_bars=8,
        )


def _panel(symbols: list[str], *, correlated: bool) -> tuple[FeaturePanel, dict]:
    """Frames whose returns are either near-identical or independent."""
    rng = np.random.default_rng(11)
    shared = rng.normal(0, 0.012, len(DATES))

    rows, index, frames = [], [], {}
    for sym in symbols:
        if correlated:
            # One factor plus a whisper of idiosyncratic noise: pairwise
            # correlation ~0.99, the "five defensives" case.
            r = shared + rng.normal(0, 0.0005, len(DATES))
        else:
            r = rng.normal(0, 0.012, len(DATES))
        close = 100 * np.exp(np.cumsum(r))
        frames[sym] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": np.full(len(DATES), 1e6)},
            index=DATES,
        )
        for d in DATES:
            rows.append(rng.normal(size=3))
            index.append((d, sym))

    X = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["date", "symbol"]),
        columns=["f0", "f1", "f2"],
    ).sort_index()
    return FeaturePanel(X=X), frames


class _StubDataset:
    def __init__(self, frames, sectors: dict[str, str] | None = None):
        from titan.core.types import AssetClass, Instrument, Universe

        self.frames = frames
        self.reliability = dict.fromkeys(frames, 1.0)
        self.benchmark_frame = next(iter(frames.values()))
        self.universe = Universe(
            instruments=[
                Instrument(
                    symbol=s,
                    asset_class=AssetClass.EQUITY,
                    sector=(sectors or {}).get(s, f"sector_{i}"),
                )
                for i, s in enumerate(frames)
            ]
        )


def _run(
    symbols: list[str],
    *,
    correlated: bool,
    cfg: TitanConfig | None = None,
    sectors: dict[str, str] | None = None,
    regime: Regime = Regime.BULL,
    size: float = 0.20,
):
    from titan.scanner.scanner import MarketScanner

    panel, frames = _panel(symbols, correlated=correlated)
    cfg = cfg or TitanConfig()
    scanner = MarketScanner(
        cfg,
        ensemble=_StubEnsemble(),
        selected_features=["f0", "f1", "f2"],
        generator=_FixedSizeGenerator(size=size),
        detector=_StubDetector(regime),
    )
    return scanner.scan(_StubDataset(frames, sectors), panel)


# --------------------------------------------------------------------------- #


def test_a_book_of_one_bet_is_sized_smaller_than_a_book_of_five():
    """The headline case: five names, one factor.

    Same requested size, same everything — only the return correlation differs.
    The correlated book must end up carrying less notional, because that is
    what the correlation penalty is for and it now runs on the live path.
    """
    syms = ["A", "B", "C", "D", "E"]
    cfg = TitanConfig()
    cfg.risk.max_sector_weight = 1.0       # isolate correlation from the sector cap
    cfg.risk.portfolio_heat_cap_pct = 100.0

    one_bet = _run(syms, correlated=True, cfg=cfg)
    five_bets = _run(syms, correlated=False, cfg=cfg)

    corr_notional = sum(s.position_size_fraction for s in one_bet.signals)
    indep_notional = sum(s.position_size_fraction for s in five_bets.signals)

    assert corr_notional < indep_notional, (
        f"correlated book took {corr_notional:.3f} of equity and the independent "
        f"book {indep_notional:.3f}: the correlation penalty is not reaching the "
        "live scanner"
    )


def test_the_heat_cap_binds_on_the_emitted_orders():
    """Summed risk-at-stop must respect the cap the config states."""
    cfg = TitanConfig()
    cfg.risk.portfolio_heat_cap_pct = 2.0
    cfg.risk.max_sector_weight = 1.0

    result = _run(["A", "B", "C", "D", "E", "F"], correlated=False, cfg=cfg)

    assert result.portfolio_heat <= cfg.risk.portfolio_heat_cap_pct + 1e-6, (
        f"emitted {result.portfolio_heat:.2f}% heat against a "
        f"{cfg.risk.portfolio_heat_cap_pct}% cap"
    )


def test_the_sector_cap_binds_across_the_emitted_orders():
    """Five names in one sector cannot each take a full-size position."""
    syms = ["A", "B", "C", "D", "E"]
    cfg = TitanConfig()
    cfg.risk.max_sector_weight = 0.30
    cfg.risk.portfolio_heat_cap_pct = 100.0

    result = _run(
        syms, correlated=False, cfg=cfg, sectors=dict.fromkeys(syms, "healthcare")
    )

    total = sum(s.position_size_fraction for s in result.signals)
    assert total <= cfg.risk.max_sector_weight + 1e-6, (
        f"{total:.3f} of equity in one sector against a "
        f"{cfg.risk.max_sector_weight} cap"
    )


def test_regime_appetite_reaches_the_live_order_sizes():
    """The generator never applies it; before this change nothing did."""
    cfg = TitanConfig()
    cfg.risk.max_sector_weight = 1.0
    cfg.risk.portfolio_heat_cap_pct = 100.0
    cfg.risk.regime_multipliers = {**cfg.risk.regime_multipliers, "bear": 0.40}

    bull = _run(["A", "B"], correlated=False, cfg=cfg, regime=Regime.BULL)
    bear = _run(["A", "B"], correlated=False, cfg=cfg, regime=Regime.BEAR)

    bull_notional = sum(s.position_size_fraction for s in bull.signals)
    bear_notional = sum(s.position_size_fraction for s in bear.signals)

    assert bear.signals, "bear regime should still trade, just smaller"
    assert bear_notional < bull_notional, (
        f"bear book {bear_notional:.3f} vs bull {bull_notional:.3f}: the regime "
        "multiplier is not reaching live sizing"
    )


def test_resizing_preserves_leverage_and_the_liquidation_price():
    """Shrinking notional must not quietly re-lever the position.

    Margin and risk are linear in notional; leverage and the liquidation level
    are ratios of it. Getting this wrong would move a stop's relationship to
    the liquidation price, which is the one invariant leverage has.
    """
    cfg = TitanConfig()
    cfg.risk.portfolio_heat_cap_pct = 1.5      # force a resize
    cfg.risk.max_sector_weight = 1.0

    result = _run(["A", "B", "C", "D"], correlated=False, cfg=cfg)

    assert result.signals
    # Without an actual resize this test proves nothing, so require one.
    assert any(s.position_size_fraction < 0.20 - 1e-9 for s in result.signals), (
        "no position was resized; the invariant below is vacuous"
    )
    for s in result.signals:
        assert s.leverage == 2.0
        assert s.liquidation_price == 80.0
        assert np.isclose(s.margin_fraction, s.position_size_fraction / s.leverage)
        # risk% was 100 * size * 4/100 at entry 100 / stop 96
        assert np.isclose(s.risk_percentage, 100.0 * s.position_size_fraction * 0.04)


def test_rejected_orders_say_the_portfolio_was_the_reason():
    """A dropped candidate must not look like a gate rejection."""
    cfg = TitanConfig()
    cfg.risk.max_sector_weight = 0.10          # room for ~one position
    cfg.risk.portfolio_heat_cap_pct = 100.0
    syms = ["A", "B", "C", "D", "E"]

    result = _run(
        syms, correlated=False, cfg=cfg, sectors=dict.fromkeys(syms, "healthcare")
    )

    statuses = [r.status for r in result.rows]
    assert any("portfolio risk" in s for s in statuses), statuses


def test_returns_are_differenced_before_alignment_not_after():
    """A post-gap return must survive into the correlation matrix.

    Aligning closes on a union index and differencing afterwards makes an
    equity's Monday read back to a NaN weekend row, so it becomes NaN too.
    That silently deletes ~19% of equity observations, and specifically the
    weekend-gap moves — the ones where correlated names move together hardest,
    and therefore the ones the correlation penalty most needs.
    """
    from titan.scanner.scanner import MarketScanner

    eq = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=60), tz="UTC")
    cr = pd.DatetimeIndex(pd.date_range("2024-01-01", periods=84, freq="D", tz="UTC"))
    rng = np.random.default_rng(5)
    frames = {}
    for sym, idx in [("AAPL", eq), ("BTC-USD", cr)]:
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))
        frames[sym] = pd.DataFrame({"close": close}, index=idx)

    scanner = MarketScanner(
        TitanConfig(), ensemble=_StubEnsemble(), selected_features=["f0"],
        generator=_FixedSizeGenerator(), detector=_StubDetector(),
    )
    returns = scanner._returns_wide(_StubDataset(frames))

    # One NaN per symbol (its first bar), and not one more.
    assert int(returns["AAPL"].notna().sum()) == len(eq) - 1
    assert int(returns["BTC-USD"].notna().sum()) == len(cr) - 1


def test_an_unmeasurable_correlation_is_sized_as_correlated_not_as_diversifying():
    """"Not measured" and "uncorrelated" must not collapse to the same number."""
    from titan.backtest.engine import PortfolioSnapshot, TradePlan
    from titan.core.config import RiskConfig
    from titan.risk.portfolio import RiskEngine

    idx = pd.DatetimeIndex(pd.bdate_range("2024-01-01", periods=80), tz="UTC")
    rng = np.random.default_rng(7)
    returns = pd.DataFrame(
        {"HELD": rng.normal(0, 0.01, len(idx)), "NEW": rng.normal(0, 0.01, len(idx))},
        index=idx,
    )
    # NEW has only three observations overlapping the window.
    returns.loc[idx[:-3], "NEW"] = np.nan

    cfg = RiskConfig(max_position_weight=1.0, portfolio_heat_cap_pct=100.0,
                     max_sector_weight=1.0)
    engine = RiskEngine(cfg, returns=returns)
    plan = TradePlan(symbol="NEW", decision_date=idx[-1], size_fraction=0.10,
                     stop_price=96.0, tp_price=106.0, max_holding_bars=5,
                     entry_ref=100.0)
    snapshot = PortfolioSnapshot(
        equity=1.0, n_positions=1, gross_exposure=0.1, open_risk_fraction=0.0,
        symbol_weights={"HELD": 0.1}, sector_weights={}, strategy_drawdown=0.0,
    )

    approved = engine.approve(plan, snapshot)

    # Full correlation penalty: half the requested size, not all of it.
    assert approved == pytest.approx(0.05)


def _vendor_frames():
    """Exactly what yfinance hands back: equities in exchange-local tz, coins in UTC."""
    eq = pd.DatetimeIndex(pd.bdate_range("2026-05-01", periods=60)).tz_localize(
        "America/New_York"
    )
    cr = pd.DatetimeIndex(
        pd.date_range("2026-05-01", periods=84, freq="D")
    ).tz_localize("UTC")
    rng = np.random.default_rng(0)
    return {
        sym: pd.DataFrame(
            {"close": 100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx))))}, index=idx
        )
        for sym, idx in [("PG", eq), ("LLY", eq), ("TRX-USD", cr)]
    }


def test_equity_and_crypto_daily_bars_land_on_the_same_calendar_grid():
    """A US equity's daily bar is stamped 00:00 New York; a coin's is 00:00 UTC.

    Concatenated raw, those are 04:00Z and 00:00Z — different instants that
    merge on nothing. The union index came out ~2x too long, every tail(window)
    covered half the history it claimed, and no equity/crypto pair shared a
    single observation, so cross-asset correlation was undefined rather than
    weak. A daily bar denotes a session, not a moment; the grid is the date.
    """
    from titan.risk.portfolio import build_returns_matrix

    frames = _vendor_frames()
    returns = build_returns_matrix(frames, "1d")
    window = returns.tail(63)

    def overlap(a: str, b: str) -> int:
        return int((window[a].notna() & window[b].notna()).sum())

    # 84 distinct calendar dates, not 60 + 84 unmerged rows.
    assert len(returns) == 84
    assert overlap("PG", "TRX-USD") > 40, "cross-asset correlation still unmeasurable"
    assert overlap("PG", "LLY") > 40, "same-class window still halved by the bloat"


def test_intraday_bars_are_not_collapsed_onto_the_day():
    """The date grid is right for sessions and destroys intraday observations."""
    from titan.risk.portfolio import build_returns_matrix

    idx = pd.DatetimeIndex(
        pd.date_range("2026-05-01 13:30", periods=40, freq="h", tz="UTC")
    )
    frames = {"X": pd.DataFrame({"close": np.linspace(100, 110, len(idx))}, index=idx)}

    assert len(build_returns_matrix(frames, "1h")) == len(idx)


def test_a_naive_decision_date_can_still_slice_a_tz_aware_matrix():
    """The matrix is normalised to UTC; `when` arrives however the vendor stamped it."""
    from titan.backtest.engine import PortfolioSnapshot, TradePlan
    from titan.core.config import RiskConfig
    from titan.risk.portfolio import RiskEngine, build_returns_matrix

    returns = build_returns_matrix(_vendor_frames(), "1d")
    engine = RiskEngine(
        RiskConfig(max_position_weight=1.0, portfolio_heat_cap_pct=100.0,
                   max_sector_weight=1.0),
        returns=returns,
    )
    plan = TradePlan(symbol="PG", decision_date=pd.Timestamp("2026-08-01"),
                     size_fraction=0.10, stop_price=96.0, tp_price=106.0,
                     max_holding_bars=5, entry_ref=100.0)
    snapshot = PortfolioSnapshot(
        equity=1.0, n_positions=1, gross_exposure=0.1, open_risk_fraction=0.0,
        symbol_weights={"TRX-USD": 0.1}, sector_weights={}, strategy_drawdown=0.0,
    )

    assert engine.approve(plan, snapshot) > 0
