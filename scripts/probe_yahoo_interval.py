#!/usr/bin/env python3
"""Report what Yahoo actually serves for a symbol at a given interval.

Data quality failures at intraday intervals are vendor-specific and change
without notice, so guessing at them from a QC score wastes runs. This prints
the facts a config decision needs: how many bars arrive, what calendar span
they cover, what fraction carry no volume, and — when volume is missing — the
pattern of WHICH bars are missing it, since a regular pattern can often be
resampled away while a random one cannot.

    python scripts/probe_yahoo_interval.py BTC-USD 1h
    python scripts/probe_yahoo_interval.py MU 1h SPY 1h BTC-USD 1d
"""

from __future__ import annotations

import sys

import pandas as pd

from titan.core.timeframe import YAHOO_MAX_HISTORY_DAYS


def probe(symbol: str, interval: str) -> None:
    import yfinance as yf

    cap = YAHOO_MAX_HISTORY_DAYS.get(interval, 0)
    period = f"{cap}d" if cap else "10y"
    df = yf.Ticker(symbol).history(period=period, interval=interval, auto_adjust=True)

    print(f"\n=== {symbol} @ {interval} (requested {period}) ===")
    if df is None or df.empty:
        print("  NO DATA RETURNED")
        return

    df.columns = [str(c).lower() for c in df.columns]
    span_days = (df.index[-1] - df.index[0]).total_seconds() / 86400.0
    print(f"  bars           {len(df)}")
    print(f"  span           {span_days:.0f} calendar days "
          f"({df.index[0].date()} -> {df.index[-1].date()})")
    print(f"  bars/day       {len(df) / max(span_days, 1):.2f}")

    if "volume" not in df:
        print("  volume         COLUMN ABSENT")
        return

    zero = df["volume"] <= 0
    print(f"  zero-volume    {zero.mean():.1%}   (QC refuses above 50%)")

    if zero.mean() > 0.02:
        # A regular pattern (e.g. every other bar, or the same hours daily) can
        # be resampled away. A uniform smear across all hours cannot.
        by_hour = zero.groupby(df.index.hour).mean()
        worst = by_hour[by_hour > 0.5]
        print(f"  hours >50% zero  {sorted(worst.index.tolist())}")
        alternating = zero.to_numpy()[:-1] != zero.to_numpy()[1:]
        print(f"  alternates       {alternating.mean():.0%} of consecutive bars differ")
        print("  by hour:")
        for hour, frac in by_hour.items():
            bar = "#" * int(frac * 40)
            print(f"    {hour:02d}:00  {frac:5.1%} {bar}")


def main(argv: list[str]) -> int:
    args = argv[1:]
    if not args:
        print(__doc__)
        return 2
    if len(args) % 2:
        print("usage: probe_yahoo_interval.py SYMBOL INTERVAL [SYMBOL INTERVAL ...]")
        return 2
    pd.set_option("display.width", 120)
    for symbol, interval in zip(args[0::2], args[1::2]):
        try:
            probe(symbol, interval)
        except Exception as exc:  # one bad symbol must not abort the batch
            print(f"\n=== {symbol} @ {interval} ===\n  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
