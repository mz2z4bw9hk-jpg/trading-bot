# Drop chart exports here

This folder is TITAN's data inbox. Put daily-bar CSV files here — one per
instrument — and the platform takes care of the rest.

## How to add files (no tools needed, phone or laptop)

1. Open this folder on github.com and click **Add file → Upload files**.
2. Drag the CSVs in and press **Commit changes**.

## What to export

- **TradingView**: open the chart, set timeframe to **1D**, chart menu →
  **Export chart data…**. Upload the file exactly as downloaded — names
  like `BINANCE_BTCUSDT, 1D.csv` are understood without renaming.
- **Any broker/platform**: any CSV with a date/time column and
  open/high/low/close/volume columns works (ISO dates or epoch seconds/ms).
  Name it `{SYMBOL}.csv` if it isn't a TradingView export.

## Rules of thumb

- Daily bars only (this platform reasons in daily bars).
- As much history as you can export — fewer than ~1500 rows will be
  flagged, and very short files are refused by the validation floor.
- Include a benchmark file (SPY or an equivalent broad index ETF).
- Prefer ETFs (SPY/QQQ/DIA) over raw indices: a volume column is required.

`configs/live-csv.yaml` is pre-wired to this folder. After uploading,
run — or ask your Claude session to run:

    titan validate --config configs/live-csv.yaml
    titan scan     --config configs/live-csv.yaml

(Files here are market data you chose to commit to your own repository.)
