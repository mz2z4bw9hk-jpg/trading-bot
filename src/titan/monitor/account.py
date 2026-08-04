"""Forward paper account: what the signals would have done to real money.

``titan track`` answers "was the model right?" — hit rate, Brier, calibration
drift. It says nothing about an account: no position sizes, no dollars, no
equity curve. This module is the other half. It replays the paper-tracking log
as a ledger against a starting balance, so the daily loop produces a running
account instead of a scorecard.

WHAT THIS IS. A faithful replay of signals *as they were sized when emitted*.
Each entry takes ``position_size_fraction`` of the equity standing at that
moment; each exit realizes the triple-barrier outcome the tracker already
graded, net of the round-trip cost the signal itself priced in. Equity
compounds on close.

WHAT THIS IS NOT. A second risk engine. Position sizes were decided at signal
time by the risk engine (¼-Kelly ∧ vol-target ∧ stop-risk) and are taken as
given here; the only portfolio-level constraint re-applied is gross exposure,
because an account that cannot fund a position simply does not take it, and
silently levering past 100% would make the equity curve fiction. Signals
skipped for that reason are recorded, not dropped.

It is also not a backtest. The backtest engine simulates a strategy over
history with intrabar fills and slippage; this replays what the live scanner
actually emitted, forward, one bar at a time as reality arrived.

MARK TO MARKET. Given price frames, open positions are additionally valued at
the latest bar and the account reports a second balance beside the realized
one, exactly as a broker does: *cash* is what has been banked by closed trades,
*account value* is cash plus the floating P&L of everything still open. Only
cash compounds into position sizing — a position sized off a paper gain that
has not been realized is sizing off an opinion. The marked figure is what the
account is worth right now; the realized figure is what it has actually earned.

The mark also decides something the tracking log cannot. A levered position
whose price passed its liquidation level did not stay open until its barrier —
the exchange closed it on the way. So the mark walks each open position's
low/high since entry and, if the level was touched, realizes the loss there
rather than waiting for a resolution that describes a trade nobody still held.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from titan.core.log import get_logger

logger = get_logger(__name__)

DEFAULT_STARTING_EQUITY = 1_000_000.0


@dataclass(slots=True)
class LedgerRow:
    """One closed round trip."""

    symbol: str
    side: str
    source: str
    asset_class: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    exit_reason: str          # tp | sl | time | liquidated
    bars_held: int
    notional: float           # dollars of exposure at entry
    margin: float             # dollars of cash actually posted
    leverage: float
    gross_return: float       # fraction, before costs
    cost: float               # fraction, round trip
    net_return: float         # fraction, after costs
    pnl: float                # dollars
    equity_after: float
    liquidated: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "side": self.side,
            "source": self.source,
            "asset_class": self.asset_class,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "entry_price": round(self.entry_price, 8),
            "exit_price": round(self.exit_price, 8),
            "exit_reason": self.exit_reason,
            "bars_held": self.bars_held,
            "notional": round(self.notional, 2),
            "margin": round(self.margin, 2),
            "leverage": round(self.leverage, 2),
            "gross_return": round(self.gross_return, 5),
            "cost": round(self.cost, 5),
            "net_return": round(self.net_return, 5),
            "pnl": round(self.pnl, 2),
            "equity_after": round(self.equity_after, 2),
            "liquidated": self.liquidated,
            # Return on the cash the position actually tied up. On a levered
            # trade this is the number that matters and it is not net_return.
            "return_on_margin": round(self.pnl / self.margin, 5) if self.margin else None,
        }


@dataclass(slots=True)
class OpenPosition:
    symbol: str
    side: str
    source: str
    asset_class: str
    entry_date: str
    entry_price: float
    notional: float
    margin: float
    leverage: float
    stop_loss: float | None
    liquidation_price: float | None = None
    take_profit_levels: list[float] = field(default_factory=list)
    cost: float = 0.0                      # round-trip fraction, owed on exit
    # Mark to market, filled in when price frames are available.
    mark_price: float | None = None
    mark_date: str | None = None

    def price_return(self, price: float) -> float:
        """Move since entry, signed by side."""
        if self.entry_price <= 0:
            return 0.0
        r = (price - self.entry_price) / self.entry_price
        return r if self.side != "short" else -r

    def unrealized(self, price: float) -> float:
        """Dollars, net of the round trip this position still owes.

        Charging the full round-trip cost against an open position means a
        freshly opened one shows a small loss immediately. That is not an
        artifact — it is what the account is worth if closed right now, and it
        is the number an operator deciding whether to hold should see.
        """
        return self.notional * (self.price_return(price) - self.cost)

    def to_dict(self) -> dict:
        d = {
            "symbol": self.symbol,
            "side": self.side,
            "source": self.source,
            "asset_class": self.asset_class,
            "entry_date": self.entry_date,
            "entry_price": round(self.entry_price, 8),
            "notional": round(self.notional, 2),
            "margin": round(self.margin, 2),
            "leverage": round(self.leverage, 2),
            "stop_loss": None if self.stop_loss is None else round(self.stop_loss, 8),
            "liquidation_price": (
                None if self.liquidation_price is None else round(self.liquidation_price, 8)
            ),
            "take_profit_levels": [round(t, 8) for t in self.take_profit_levels],
            "mark_price": None,
            "mark_date": self.mark_date,
            "price_return": None,
            "unrealized_pnl": None,
            "return_on_margin": None,
            "distance_to_stop": None,
            "distance_to_liquidation": None,
        }
        if self.mark_price is None:
            return d
        price = self.mark_price
        pnl = self.unrealized(price)
        d["mark_price"] = round(price, 8)
        d["price_return"] = round(self.price_return(price), 5)
        d["unrealized_pnl"] = round(pnl, 2)
        d["return_on_margin"] = round(pnl / self.margin, 5) if self.margin else None
        if self.stop_loss:
            d["distance_to_stop"] = round((price - self.stop_loss) / price, 5)
        if self.liquidation_price:
            d["distance_to_liquidation"] = round(
                abs(price - self.liquidation_price) / price, 5
            )
        return d


def _simple_return(log_return: float) -> float:
    """The labeller records log returns; an account compounds simple ones."""
    return math.expm1(log_return)


def _liquidation_distance(rec: dict, entry: float) -> float | None:
    """How far the recorded liquidation level sits from the entry, as a fraction.

    Taken from the level the order was *written with*, not recomputed from
    today's config: leverage settings can change between the scan that opened a
    position and the resolve that grades it, and re-deriving the level here
    would quietly rewrite where a past trade would have died.
    """
    liq = rec.get("liquidation_price")
    if liq is None or entry <= 0:
        return None
    d = abs(entry - float(liq)) / entry
    return d if d > 0 else None


def _daily_marks(frames: Mapping[str, Any] | None) -> dict[str, dict[str, dict[str, float]]]:
    """Per symbol, per calendar date: the day's close, low and high.

    Keyed by ``YYYY-MM-DD`` strings because that is what the tracking log
    records — it stores ``str(signal.date.date())``, so an intraday config
    already collapses to daily there and the mark has to meet it on the same
    grid. The day's last bar supplies the close; low and high span every bar in
    the day, which is what a liquidation touch has to be tested against.
    """
    if not frames:
        return {}
    out: dict[str, dict[str, dict[str, float]]] = {}
    for symbol, frame in frames.items():
        if frame is None or len(frame) == 0:
            continue
        try:
            grouped = frame.groupby(frame.index.date)
            agg = grouped.agg({"close": "last", "low": "min", "high": "max"})
        except (KeyError, AttributeError, TypeError):
            continue
        out[str(symbol)] = {
            str(day): {
                "close": float(row.close), "low": float(row.low), "high": float(row.high),
            }
            for day, row in agg.iterrows()
            if math.isfinite(row.close)
        }
    return out


def _liquidation_touched(pos: OpenPosition, bar: dict[str, float]) -> bool:
    """Did this bar reach the position's liquidation level?

    A long dies on the low, a short on the high — the extreme, not the close.
    Tested bar by bar as the replay walks forward, so a position that traded
    through its liquidation and recovered is still closed at the time it
    happened; an account that only checked where price ended up would report it
    as a live winner.
    """
    if pos.liquidation_price is None or pos.leverage <= 1.0:
        return False
    if pos.side == "short":
        return bar["high"] >= pos.liquidation_price
    return bar["low"] <= pos.liquidation_price


def replay(
    records: list[dict],
    *,
    starting_equity: float = DEFAULT_STARTING_EQUITY,
    max_gross_exposure: float = 1.0,
    max_account_leverage: float = 1.0,
    frames: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Replay tracked predictions into an account state.

    ``records`` are :class:`~titan.monitor.paper.PaperTrackingStore` rows.
    Rows logged before the account fields existed (no ``size_fraction``) are
    counted and skipped rather than guessed at — inventing a size would put
    fabricated dollars on the equity curve.

    Two independent ceilings apply on entry, because leverage separates two
    things that are identical in a cash account. ``max_gross_exposure`` bounds
    the *cash* posted as margin — an account cannot fund what it does not have.
    ``max_account_leverage`` bounds the summed *notional* as a multiple of
    equity — an account that has funded ten 3x positions has 30x of market
    exposure behind one balance, and no per-position limit sees that.

    ``frames`` turns on mark to market: open positions are valued at the latest
    bar, the equity curve gains a point per day instead of one per close, and
    levered positions whose price reached their liquidation level are realized
    there. Without it the replay behaves exactly as before — realized-only,
    which is correct, just blind between closes.
    """
    events: list[tuple[str, int, dict]] = []
    legacy = 0
    for rec in records:
        if rec.get("size_fraction") is None:
            legacy += 1
            continue
        events.append((rec["date"], 0, rec))  # 0 sorts opens before closes
        if rec.get("outcome") is not None and rec.get("exit_date"):
            events.append((rec["exit_date"], 1, rec))
    events.sort(key=lambda e: (e[0], e[1]))

    equity = float(starting_equity)
    peak = equity
    max_drawdown = 0.0
    margin_used = 0.0                             # cash posted, dollars
    gross_notional = 0.0                          # exposure, dollars
    peak_account_leverage = 0.0
    live: dict[tuple[str, str], OpenPosition] = {}
    ledger: list[LedgerRow] = []
    equity_curve: list[dict] = []
    skipped: list[dict] = []
    n_liquidated = 0

    marks = _daily_marks(frames)
    events_by_date: dict[str, list[tuple[int, dict]]] = {}
    for date, kind, rec in events:
        events_by_date.setdefault(date, []).append((kind, rec))

    # The timeline is every day something can happen. Without marks that is
    # only the event dates — the account is blind between them by construction.
    # With marks it is every trading day from the first entry onward, which is
    # what lets the balance move on days no trade opened or closed.
    timeline = set(events_by_date)
    if marks and events_by_date:
        first = min(events_by_date)
        for per_symbol in marks.values():
            timeline.update(d for d in per_symbol if d >= first)
    dates = sorted(timeline)
    date_index = {d: i for i, d in enumerate(dates)}
    last_price: dict[str, float] = {}
    mark_date: str | None = None
    peak_marked = equity
    max_marked_drawdown = 0.0

    def _open(date: str, rec: dict) -> None:
        nonlocal margin_used, gross_notional, peak_account_leverage
        notional = equity * float(rec["size_fraction"])
        leverage = float(rec.get("leverage") or 1.0)
        margin = notional / leverage if leverage > 0 else notional
        if margin_used + margin > equity * max_gross_exposure + 1e-9:
            skipped.append({
                "symbol": rec["symbol"], "date": rec["date"],
                "reason": "would exceed gross exposure cap",
            })
            return
        if gross_notional + notional > equity * max_account_leverage + 1e-9:
            skipped.append({
                "symbol": rec["symbol"], "date": rec["date"],
                "reason": f"would exceed account leverage cap ({max_account_leverage:.1f}x)",
            })
            return
        margin_used += margin
        gross_notional += notional
        peak_account_leverage = max(peak_account_leverage, gross_notional / equity)
        live[(rec["symbol"], rec["date"])] = OpenPosition(
            symbol=rec["symbol"],
            side=rec.get("side", "long"),
            source=str(rec.get("source") or "model"),
            asset_class=str(rec.get("asset_class") or "equity"),
            entry_date=rec["date"],
            # The realized fill is only known once the trade resolves; until
            # then the signal's own expected entry is what the position was
            # opened against, and an open row with price 0 is useless.
            entry_price=float(rec.get("entry_price") or rec.get("signal_entry") or 0.0),
            notional=notional,
            margin=margin,
            leverage=leverage,
            stop_loss=rec.get("stop_loss"),
            liquidation_price=rec.get("liquidation_price"),
            take_profit_levels=list(rec.get("take_profit_levels") or []),
            cost=float(rec.get("cost_estimate") or 0.0),
        )

    def _book(
        pos: OpenPosition, date: str, *, gross_ret: float, cost: float,
        exit_price: float, exit_reason: str, bars_held: int, forced: bool = False,
    ) -> None:
        """Realize a position: release its margin, post its P&L, compound."""
        nonlocal equity, peak, max_drawdown, margin_used, gross_notional, n_liquidated
        margin_used -= pos.margin
        gross_notional -= pos.notional
        net_ret = gross_ret - cost
        pnl = pos.notional * net_ret

        # A levered position cannot lose more than the margin behind it: the
        # exchange closes it first. Without this the ledger would post a 3x
        # position down 50% as a 150% loss of notional, which is not a number
        # any account can produce.
        liquidated = forced or (pos.leverage > 1.0 and pnl < -pos.margin)
        if liquidated:
            pnl = -pos.margin
            n_liquidated += 1

        equity += pnl
        peak = max(peak, equity)
        max_drawdown = min(max_drawdown, equity / peak - 1.0)
        ledger.append(LedgerRow(
            symbol=pos.symbol, side=pos.side, source=pos.source,
            asset_class=pos.asset_class,
            entry_date=pos.entry_date, exit_date=date,
            entry_price=pos.entry_price, exit_price=exit_price,
            exit_reason="liquidated" if liquidated else exit_reason,
            bars_held=bars_held,
            notional=pos.notional, margin=pos.margin, leverage=pos.leverage,
            gross_return=gross_ret, cost=cost,
            net_return=net_ret, pnl=pnl, equity_after=equity,
            liquidated=liquidated,
        ))

    for date in dates:
        for kind, rec in sorted(events_by_date.get(date, []), key=lambda e: e[0]):
            if kind == 0:
                _open(date, rec)
                continue
            pos = live.pop((rec["symbol"], rec["date"]), None)
            if pos is None:  # its open was skipped, or it liquidated en route
                continue
            gross_ret = _simple_return(float(rec["ret"]))
            cost = float(rec.get("cost_estimate") or 0.0)
            d_liq = _liquidation_distance(rec, pos.entry_price)
            _book(
                pos, date,
                gross_ret=gross_ret, cost=cost,
                exit_price=float(rec.get("exit_price") or 0.0),
                exit_reason=str(rec.get("touch") or "?"),
                bars_held=int(rec.get("bars_held") or 0),
                forced=(
                    pos.leverage > 1.0 and d_liq is not None
                    and gross_ret - cost <= -d_liq
                ),
            )
            if not marks:
                equity_curve.append({"date": date, "equity": round(equity, 2)})

        if not marks:
            continue

        # ---- mark to market --------------------------------------------
        for symbol, per_symbol in marks.items():
            bar = per_symbol.get(date)
            if bar is not None:
                last_price[symbol] = bar["close"]

        # A levered position that reached its liquidation level today is gone,
        # whatever the tracking log later says its barrier outcome was. Checked
        # from the day AFTER entry: the entry bar's low precedes the fill.
        for key, pos in list(live.items()):
            if pos.entry_date >= date:
                continue
            bar = marks.get(pos.symbol, {}).get(date)
            if bar is None or not _liquidation_touched(pos, bar):
                continue
            del live[key]
            _book(
                pos, date,
                gross_ret=-1.0 / pos.leverage, cost=pos.cost,
                exit_price=pos.liquidation_price or 0.0,
                exit_reason="liquidated",
                bars_held=date_index[date] - date_index.get(pos.entry_date, 0),
                forced=True,
            )

        unrealized = 0.0
        for pos in live.values():
            price = last_price.get(pos.symbol)
            if price is None:
                continue
            pos.mark_price, pos.mark_date = price, date
            unrealized += pos.unrealized(price)

        mark_date = date
        account_value = equity + unrealized
        peak_marked = max(peak_marked, account_value)
        max_marked_drawdown = min(max_marked_drawdown, account_value / peak_marked - 1.0)
        equity_curve.append({
            "date": date,
            "equity": round(account_value, 2),   # what the account is worth
            "cash": round(equity, 2),            # what it has actually banked
            "unrealized": round(unrealized, 2),
        })

    def _bucket(key: Callable[[LedgerRow], str]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for row in ledger:
            b = out.setdefault(key(row), {"n": 0, "pnl": 0.0, "wins": 0})
            b["n"] += 1
            b["pnl"] += row.pnl
            b["wins"] += 1 if row.pnl > 0 else 0
        for b in out.values():
            b["pnl"] = round(b["pnl"], 2)
            b["win_rate"] = round(b["wins"] / b["n"], 4) if b["n"] else None
        return out

    # Which engine actually earned: the whole reason technical orders are
    # labelled rather than merged into one undifferentiated stream. Same
    # question by asset class, now that the two books trade on different terms.
    by_source = _bucket(lambda r: r.source)
    by_asset_class = _bucket(lambda r: r.asset_class)

    wins = [r for r in ledger if r.pnl > 0]
    losses = [r for r in ledger if r.pnl <= 0]
    gross_profit = sum(r.pnl for r in wins)
    gross_loss = -sum(r.pnl for r in losses)

    if legacy:
        logger.info(
            "paper account: %d prediction(s) logged before account tracking "
            "existed have no size and are excluded", legacy,
        )

    unrealized_now = sum(
        p.unrealized(p.mark_price) for p in live.values() if p.mark_price is not None
    )
    account_value = equity + unrealized_now
    n_unmarked = sum(1 for p in live.values() if p.mark_price is None)
    if n_unmarked and marks:
        logger.info(
            "paper account: %d open position(s) have no price in the dataset and "
            "are carried at cost", n_unmarked,
        )

    return {
        "starting_equity": round(starting_equity, 2),
        # ``equity`` stays the REALIZED balance — it is what sizes positions,
        # and sizing off an unrealized gain is sizing off an opinion. The
        # marked figures sit beside it, the way a broker separates cash from
        # net liquidation value.
        "equity": round(equity, 2),
        "total_return": round(equity / starting_equity - 1.0, 5),
        "realized_pnl": round(equity - starting_equity, 2),
        "max_drawdown": round(max_drawdown, 5),
        "account_value": round(account_value, 2),
        "unrealized_pnl": round(unrealized_now, 2),
        "marked_return": round(account_value / starting_equity - 1.0, 5),
        "max_marked_drawdown": round(max_marked_drawdown, 5) if marks else None,
        "mark_date": mark_date,
        "n_unmarked": n_unmarked,
        "n_closed": len(ledger),
        "n_open": len(live),
        "open_notional": round(sum(p.notional for p in live.values()), 2),
        "open_margin": round(sum(p.margin for p in live.values()), 2),
        "account_leverage": round(gross_notional / equity, 3) if equity > 0 else None,
        "peak_account_leverage": round(peak_account_leverage, 3),
        "max_account_leverage": max_account_leverage,
        "n_liquidated": n_liquidated,
        "win_rate": round(len(wins) / len(ledger), 4) if ledger else None,
        "avg_win": round(sum(r.net_return for r in wins) / len(wins), 5) if wins else None,
        "avg_loss": round(sum(r.net_return for r in losses) / len(losses), 5) if losses else None,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else None,
        "by_source": by_source,
        "by_asset_class": by_asset_class,
        "n_legacy_unsized": legacy,
        "n_skipped_exposure": len(skipped),
        "skipped": skipped[-20:],
        "equity_curve": equity_curve,
        # Biggest floating loser first: what an operator wants to see on top of
        # an open book is what is hurting, not what happens to sort first.
        "open_positions": [
            p.to_dict() for p in sorted(
                live.values(),
                key=lambda p: p.unrealized(p.mark_price) if p.mark_price is not None else 0.0,
            )
        ],
        # Newest first: the dashboard shows the most recent day's trades on top.
        "trades": [r.to_dict() for r in reversed(ledger)],
    }
