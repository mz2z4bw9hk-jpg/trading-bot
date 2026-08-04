"""TITAN command-line interface.

Commands
--------
- ``titan validate``  full walk-forward validation; writes artifacts; saves the
  final-fold model bundle to the registry and runs the champion/challenger
  promotion gate against any existing production model.
- ``titan scan``      rank the universe on the latest bar with the production
  bundle; writes scan artifacts.
- ``titan dashboard`` serve the dashboard + JSON API over the artifacts dir.
- ``titan export``    write the dashboard + artifacts as ONE self-contained
  HTML file: open by double-click, share, or drop on any static host — no
  server, no Python needed to view it.
- ``titan track``     paper-tracking: ``resolve`` grades logged scan
  predictions against what the market actually did (scans log themselves);
  ``status`` prints live calibration + the CUSUM decay alarm.
- ``titan preflight`` QC a universe without running research: which symbols
  would survive and why not. Minutes instead of the hours a large run costs.
- ``titan account``   forward paper account replayed from the tracking log:
  starting balance, equity, open positions and the closed-trade ledger.
- ``titan info``      show config, registry, tracking and artifact status.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from titan import __version__
from titan.core.config import TitanConfig, bar_clock, load_config, research_fingerprint
from titan.core.jsonsafe import json_safe
from titan.core.log import configure_logging, get_logger

logger = get_logger(__name__)

DEFAULT_CONFIG = Path("configs/default.yaml")


def _load_cfg(args: argparse.Namespace) -> TitanConfig:
    path = Path(args.config) if args.config else (DEFAULT_CONFIG if DEFAULT_CONFIG.exists() else None)
    overrides: dict = {}
    if getattr(args, "folds", None):
        overrides.setdefault("cv", {})["n_folds"] = args.folds
    if getattr(args, "tuning", None) is not None:
        overrides.setdefault("model", {})["tuning_iterations"] = args.tuning
    if getattr(args, "seed", None) is not None:
        overrides.setdefault("run", {})["seed"] = args.seed
    cfg = load_config(path, overrides)
    configure_logging(cfg.run.log_level)
    return cfg


# --------------------------------------------------------------------- #

def cmd_validate(args: argparse.Namespace) -> int:
    from titan.artifacts import write_scan_artifacts, write_walkforward_artifacts
    from titan.backtest.costs import CostModel
    from titan.backtest.walkforward import WalkForwardRunner
    from titan.data.store import MarketDataStore
    from titan.features.pipeline import FeatureMatrixBuilder
    from titan.models.registry import ModelRegistry
    from titan.monitor.compare import compare_returns, promotion_gate
    from titan.regime.detector import RegimeDetector
    from titan.scanner.scanner import MarketScanner
    from titan.signals.generator import SignalGenerator

    cfg = _load_cfg(args)
    out_dir = Path(args.out or cfg.run.artifacts_dir)

    logger.info("TITAN validate: provider=%s seed=%d", cfg.data.provider, cfg.run.seed)
    if cfg.risk.leverage.max_leverage:
        # Stated once, loudly, rather than left for someone to infer from a
        # Sharpe that looks lower than the live book's. The walk-forward is the
        # evidence base for the MODEL; simulating margin through it would need
        # intrabar liquidation and per-bar funding, and a half-modelled version
        # would report returns the account could not have produced.
        logger.warning(
            "risk.leverage is configured (%s) but the walk-forward runs UNLEVERED. "
            "Backtest Sharpe, CAGR and drawdown below describe the cash strategy; "
            "live orders from `titan scan` carry the multiple and the paper "
            "account is where its effect shows up.",
            ", ".join(f"{k}={v}x" for k, v in sorted(cfg.risk.leverage.max_leverage.items())),
        )
    dataset = MarketDataStore(cfg.data, cfg.universe, seed=cfg.run.seed).load()
    builder = FeatureMatrixBuilder(cfg.features)
    panel = builder.build(dataset)

    runner = WalkForwardRunner(cfg)
    report = runner.run(dataset, panel=panel)
    write_walkforward_artifacts(out_dir, cfg, dataset, report)

    # ---- model registry + promotion gate ------------------------------
    registry = ModelRegistry(cfg.model.store_dir)
    artifacts = runner.last_fold_artifacts
    oos_returns = report.backtest.returns
    bundle = {
        "ensemble": artifacts["ensemble"],
        "selected": artifacts["selected"],
        "explainer": artifacts["explainer"],
        "analogues": artifacts["analogues"],
        "oos_returns": oos_returns,
    }
    summary = report.backtest.summary.to_dict()
    metrics = {
        "pooled_auc": report.pooled_auc,
        "pooled_brier": report.pooled_brier,
        "sharpe": summary["sharpe"],
        "max_drawdown": summary["max_drawdown"],
        "n_signals": len(report.signals),
    }
    version = registry.save(
        bundle,
        metrics=metrics,
        feature_names=artifacts["selected"],
        description=f"walk-forward {len(report.folds)} folds on {cfg.data.provider}",
        config_fingerprint=research_fingerprint(cfg),
        train_start=str(report.folds[0].test_start) if report.folds else "",
        train_end=str(report.folds[-1].test_end) if report.folds else "",
    )

    production = registry.production_version()
    if production is None:
        registry.promote(version)
        logger.info("no production model existed; %s promoted", version)
    else:
        prod_bundle, prod_manifest = registry.load(production)
        comparison = compare_returns(oos_returns, prod_bundle["oos_returns"], seed=cfg.run.seed)
        approved, reasons = promotion_gate(
            comparison, metrics, prod_manifest.metrics,
            p_value_required=cfg.monitor.promotion_p_value,
        )
        if approved:
            registry.promote(version)
        else:
            logger.info("challenger %s NOT promoted: %s", version, "; ".join(reasons))

    # ---- scan with the freshly validated bundle ------------------------
    cost_model = CostModel(cfg.backtest.costs)
    ppy = bar_clock(cfg).bars_per_year
    generator = SignalGenerator(cfg.signals, cfg.labels, cfg.risk, cost_model,
                                cfg.backtest.max_positions, periods_per_year=ppy)
    detector = RegimeDetector(cfg.regime, seed=cfg.run.seed, periods_per_year=ppy)
    bench = dataset.benchmark_frame
    detector.fit(bench.iloc[: max(len(bench) - 63, cfg.regime.min_train_bars)])
    scanner = MarketScanner(
        cfg,
        ensemble=artifacts["ensemble"],
        selected_features=artifacts["selected"],
        generator=generator,
        detector=detector,
        explainer=artifacts["explainer"],
        analogues=artifacts["analogues"],
    )
    scan = scanner.scan(dataset, panel)
    write_scan_artifacts(out_dir, scan)

    s = report.backtest.summary
    print(json.dumps({
        "pooled_auc": round(report.pooled_auc, 4),
        "high_conf_hit_rate": round(report.high_conf_hit_rate, 4),
        "base_rate": round(report.base_rate, 4),
        "oos_sharpe": round(s.sharpe, 3),
        "oos_max_drawdown": round(s.max_drawdown, 4),
        "oos_cagr": round(s.cagr, 4),
        "n_trades": s.n_trades,
        "bootstrap_sharpe_ci": [round(v, 3) for v in report.bootstrap.sharpe_ci],
        "model_version": version,
        "artifacts": str(out_dir.resolve()),
    }, indent=1))
    return 0


def _config_mismatch_error(manifest, cfg: TitanConfig, allow: bool) -> str | None:
    """Refuse to run inference when the active config describes a different
    world than the one the production model was validated on.

    A strong-market model scanned against another config's universe produces
    plausible-looking, meaningless numbers — the exact failure mode this
    platform exists to prevent. Old manifests without a fingerprint skip the
    check (nothing to compare).
    """
    fp = research_fingerprint(cfg)
    if not manifest.config_fingerprint or manifest.config_fingerprint == fp:
        return None
    if allow:
        logger.warning(
            "config mismatch overridden: model %s fingerprint %s vs active %s",
            manifest.version, manifest.config_fingerprint, fp,
        )
        return None
    return (
        f"CONFIG MISMATCH: production model {manifest.version} was validated on a different "
        f"research configuration (fingerprint {manifest.config_fingerprint}, active {fp}).\n"
        f"Numbers from a model scanned against a world it never saw are meaningless.\n"
        f"Pass the SAME --config used for `titan validate` (note: no --config means "
        f"configs/default.yaml), re-validate under this config, or override with "
        f"--allow-config-mismatch if you truly know better."
    )


def _paper_store_path(cfg: TitanConfig) -> Path:
    return Path(cfg.model.store_dir) / "paper_track.json"


def _baseline_brier(cfg: TitanConfig) -> float:
    """CUSUM baseline: the production model's own OOF Brier, else coin-flip."""
    from titan.models.registry import ModelRegistry
    from titan.monitor.paper import DEFAULT_BASELINE_BRIER

    try:
        registry = ModelRegistry(cfg.model.store_dir)
        version = registry.production_version()
        if version is None:
            return DEFAULT_BASELINE_BRIER
        _, manifest = registry.load(version)
        return float(manifest.metrics.get("pooled_brier", DEFAULT_BASELINE_BRIER))
    except Exception:
        return DEFAULT_BASELINE_BRIER


def _write_tracking_artifact(cfg: TitanConfig, out_dir: Path, summary: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tracking.json").write_text(json.dumps(summary, indent=1, default=str))


def _write_account_artifact(
    cfg: TitanConfig, out_dir: Path, frames: dict | None = None
) -> dict:
    """Refresh the forward paper account from the tracking log.

    Derived state, not a separate ledger: it is recomputed from the log every
    time the log moves, so `scan` (which opens positions) and `track resolve`
    (which closes them) both keep it current without a third command to forget.

    ``frames`` marks the open book to the latest bar. Every caller that already
    has a dataset passes it, so the balance moves with price rather than only
    when a trade resolves; ``titan account`` loads one for the same reason.
    """
    from titan.monitor.account import replay
    from titan.monitor.paper import PaperTrackingStore

    store = PaperTrackingStore(_paper_store_path(cfg))
    state = replay(
        store.records,
        starting_equity=cfg.monitor.paper_starting_equity,
        max_gross_exposure=cfg.backtest.max_gross_exposure,
        max_account_leverage=cfg.risk.leverage.max_account_leverage,
        frames=frames,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "account.json").write_text(
        json.dumps(json_safe(state), indent=1, default=str, allow_nan=False)
    )
    return state


def cmd_scan(args: argparse.Namespace) -> int:
    from titan.artifacts import write_scan_artifacts
    from titan.backtest.costs import CostModel
    from titan.data.store import MarketDataStore
    from titan.features.pipeline import FeatureMatrixBuilder
    from titan.models.registry import ModelRegistry
    from titan.monitor.paper import PaperTrackingStore
    from titan.regime.detector import RegimeDetector
    from titan.scanner.scanner import MarketScanner
    from titan.signals.generator import SignalGenerator

    cfg = _load_cfg(args)
    registry = ModelRegistry(cfg.model.store_dir)
    wanted = getattr(args, "model", None)
    try:
        bundle, manifest = registry.load(wanted)
    except LookupError as exc:
        print(
            f"model {wanted!r} not found in the registry: {exc}" if wanted
            else "no production model: run `titan validate` first",
            file=sys.stderr,
        )
        return 2
    error = _config_mismatch_error(manifest, cfg, args.allow_config_mismatch)
    if error:
        print(error, file=sys.stderr)
        return 2
    logger.info(
        "scanning with %s model %s",
        "requested" if wanted else "production", manifest.version,
    )

    dataset = MarketDataStore(cfg.data, cfg.universe, seed=cfg.run.seed).load()
    panel = FeatureMatrixBuilder(cfg.features).build(dataset)
    ppy = bar_clock(cfg).bars_per_year
    generator = SignalGenerator(
        cfg.signals, cfg.labels, cfg.risk, CostModel(cfg.backtest.costs),
        cfg.backtest.max_positions, periods_per_year=ppy,
    )
    detector = RegimeDetector(cfg.regime, seed=cfg.run.seed, periods_per_year=ppy)
    bench = dataset.benchmark_frame
    detector.fit(bench.iloc[: max(len(bench) - 63, cfg.regime.min_train_bars)])
    scanner = MarketScanner(
        cfg,
        ensemble=bundle["ensemble"],
        selected_features=bundle["selected"],
        generator=generator,
        detector=detector,
        explainer=bundle["explainer"],
        analogues=bundle["analogues"],
    )
    scan = scanner.scan(dataset, panel)
    out = write_scan_artifacts(Path(args.out or cfg.run.artifacts_dir), scan)

    # Paper-track every emitted signal; `titan track resolve` grades them later.
    store = PaperTrackingStore(_paper_store_path(cfg))
    n_tracked = store.log_signals(scan.signals)
    _write_tracking_artifact(cfg, out, store.summary(_baseline_brier(cfg)))
    account = _write_account_artifact(cfg, out, dataset.frames)

    print(json.dumps({
        "date": str(scan.date.date()),
        "account_equity": account["equity"],
        "open_positions": account["n_open"],
        "regime": scan.regime.regime.value,
        "n_signals": len(scan.signals),
        "orders_by_asset_class": {
            k: len(v) for k, v in sorted(scan.signals_by_asset_class().items())
        },
        "newly_tracked": n_tracked,
        "top": [
            {"symbol": s.symbol, "asset_class": s.asset_class,
             "grade": s.trade_grade.value, "leverage": round(s.leverage, 2),
             "confidence": round(s.confidence_score, 1)}
            for s in scan.signals[:10]
        ],
        "artifacts": str(out.resolve()),
    }, indent=1))
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    import uvicorn

    from titan.server.app import create_app

    cfg = _load_cfg(args)
    artifacts_dir = Path(args.artifacts or cfg.run.artifacts_dir)
    app = create_app(artifacts_dir)
    logger.info("dashboard on http://%s:%d (artifacts: %s)", args.host, args.port, artifacts_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    from titan.server.export import export_static_dashboard

    cfg = _load_cfg(args)
    artifacts_dir = Path(args.artifacts or cfg.run.artifacts_dir)
    out = Path(args.out or (artifacts_dir / "titan_dashboard.html"))
    try:
        path, payloads = export_static_dashboard(artifacts_dir, out)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "output": str(path.resolve()),
        "size_kb": round(path.stat().st_size / 1024, 1),
        "embedded": sorted(k for k, v in payloads.items() if v is not None),
        "missing": sorted(k for k, v in payloads.items() if v is None),
    }, indent=1))
    return 0


def cmd_track(args: argparse.Namespace) -> int:
    from titan.monitor.paper import PaperTrackingStore

    cfg = _load_cfg(args)
    store = PaperTrackingStore(_paper_store_path(cfg))

    n_resolved = 0
    frames: dict | None = None
    if args.action == "resolve":
        from titan.data.store import MarketDataStore
        from titan.models.registry import ModelRegistry

        registry = ModelRegistry(cfg.model.store_dir)
        production = registry.production_version()
        if production is not None:
            _, manifest = registry.load(production)
            error = _config_mismatch_error(manifest, cfg, args.allow_config_mismatch)
            if error:
                print(error, file=sys.stderr)
                return 2
        dataset = MarketDataStore(cfg.data, cfg.universe, seed=cfg.run.seed).load()
        frames = dataset.frames
        n_resolved = store.resolve(frames, cfg.labels)

    summary = store.summary(_baseline_brier(cfg))
    out_dir = Path(args.out or cfg.run.artifacts_dir)
    _write_tracking_artifact(cfg, out_dir, summary)
    account = _write_account_artifact(
        cfg, Path(args.out or cfg.run.artifacts_dir), frames
    )
    print(json.dumps({
        "newly_resolved": n_resolved,
        "account_equity": account["equity"],
        "account_return": account["total_return"],
        "closed_trades": account["n_closed"],
        "open_positions": account["n_open"],
        **summary,
    }, indent=1, default=str))
    if summary.get("cusum_alarm"):
        print(
            "CUSUM ALARM: live calibration is degrading — run `titan validate` "
            "to produce a challenger and let the promotion gate decide.",
            file=sys.stderr,
        )
        return 3
    return 0


def cmd_preflight(args: argparse.Namespace) -> int:
    """QC a universe without running research: which symbols survive, and why not."""
    from titan.data.store import MarketDataStore

    cfg = _load_cfg(args)
    clock = bar_clock(cfg)
    logger.info(
        "preflight: %d instruments, provider=%s, %s bars%s",
        len(cfg.universe.instruments), cfg.data.provider, cfg.data.timeframe,
        f" resampled from {cfg.data.resample_from}" if cfg.data.resample_from else "",
    )
    results = MarketDataStore(cfg.data, cfg.universe, seed=cfg.run.seed).preflight()

    passing, failing = [], []
    for symbol, result in sorted(results.items()):
        if isinstance(result, str):
            failing.append((symbol, 0, 0.0, result))
        elif result.reliability < cfg.data.min_reliability:
            failing.append(
                (symbol, result.n_bars, result.reliability, "; ".join(result.issues[:2]))
            )
        else:
            passing.append((symbol, result.n_bars, result.reliability))

    print(f"\n{'SYMBOL':12s} {'BARS':>7s} {'RELIAB':>7s}  STATUS")
    for symbol, bars, rel in passing:
        print(f"{symbol:12s} {bars:7d} {rel:7.3f}  ok")
    for symbol, bars, rel, why in failing:
        print(f"{symbol:12s} {bars:7d} {rel:7.3f}  EXCLUDED: {why[:70]}")

    fold_bars = cfg.cv.min_train_bars + cfg.cv.n_folds * cfg.cv.test_bars
    shortest = min((b for _, b, _ in passing), default=0)
    longest = max((b for _, b, _ in passing), default=0)
    print(
        f"\n{len(passing)}/{len(results)} symbols pass QC at floor "
        f"{cfg.data.min_reliability:.2f} | bars {shortest}-{longest} | "
        f"{cfg.cv.n_folds} folds need {fold_bars} | clock {clock.timeframe} "
        f"{clock.bars_per_year:.0f}/yr"
    )
    if not passing:
        print("NOTHING would run. Fix the universe or the interval before validating.")
        return 2
    if longest < fold_bars:
        print(
            f"WARNING: longest series has {longest} bars but {cfg.cv.n_folds} folds need "
            f"{fold_bars} — the run will quietly use fewer folds."
        )
    if failing:
        print("\nUniverse with the failures removed:\n")
        print("  instruments:")
        by_symbol = {i.symbol: i for i in cfg.universe.instruments}
        for symbol, _, _ in passing:
            item = by_symbol.get(symbol)
            if item is not None:
                print(
                    f"    - {{ symbol: {item.symbol}, asset_class: {item.asset_class}, "
                    f"sector: {item.sector} }}"
                )
    return 0


def cmd_account(args: argparse.Namespace) -> int:
    """Print the forward paper account: equity, open book, recent trades."""
    cfg = _load_cfg(args)

    # Marking the open book needs prices. Loading a dataset costs a cached read
    # in the ordinary case; --no-marks skips it for a purely realized view.
    frames = None
    if not getattr(args, "no_marks", False):
        from titan.data.store import MarketDataStore

        try:
            frames = MarketDataStore(cfg.data, cfg.universe, seed=cfg.run.seed).load().frames
        except (OSError, ValueError) as exc:
            logger.warning("could not load prices to mark the open book: %s", exc)

    state = _write_account_artifact(cfg, Path(args.out or cfg.run.artifacts_dir), frames)

    eq, start = state["equity"], state["starting_equity"]
    print(f"\nPAPER ACCOUNT  (replayed from {_paper_store_path(cfg)})")
    print(f"  starting equity   ${start:,.2f}")
    print(f"  cash (realized)   ${eq:,.2f}   ({state['total_return']:+.2%})")
    if state["mark_date"]:
        print(f"  account value     ${state['account_value']:,.2f}"
              f"   ({state['marked_return']:+.2%})   marked {state['mark_date']}")
        print(f"  unrealized P&L    ${state['unrealized_pnl']:,.2f}")
    print(f"  realized P&L      ${state['realized_pnl']:,.2f}")
    print(f"  max drawdown      {state['max_drawdown']:.2%}"
          + (f"   (marked {state['max_marked_drawdown']:.2%})"
             if state["max_marked_drawdown"] is not None else ""))
    print(f"  closed / open     {state['n_closed']} / {state['n_open']}"
          f"   (open notional ${state['open_notional']:,.2f})")
    if state["win_rate"] is not None:
        print(f"  win rate          {state['win_rate']:.1%}"
              f"   profit factor {state['profit_factor']}")
    if state["n_skipped_exposure"]:
        print(f"  skipped (gross exposure cap): {state['n_skipped_exposure']}")
    if state["n_legacy_unsized"]:
        print(f"  excluded (logged before account tracking): {state['n_legacy_unsized']}")

    if state["open_positions"]:
        print("\n  OPEN POSITIONS  (worst floating P&L first)")
        print(f"    {'SYMBOL':12s} {'SINCE':12s} {'ENTRY':>12s} {'MARK':>12s} "
              f"{'LEV':>5s} {'NOTIONAL':>14s} {'MOVE':>8s} {'UNREAL':>14s} {'STOP':>12s}")
        for p_ in state["open_positions"]:
            stop = "—" if p_["stop_loss"] is None else f"{p_['stop_loss']:.6g}"
            mark = "—" if p_["mark_price"] is None else f"{p_['mark_price']:.6g}"
            move = "—" if p_["price_return"] is None else f"{p_['price_return']:+.2%}"
            unreal = "—" if p_["unrealized_pnl"] is None else f"{p_['unrealized_pnl']:+,.2f}"
            lev = "—" if p_["leverage"] <= 1 else f"{p_['leverage']:.1f}x"
            print(f"    {p_['symbol']:12s} {p_['entry_date']:12s} "
                  f"{p_['entry_price']:12.6g} {mark:>12s} {lev:>5s} "
                  f"{p_['notional']:14,.2f} {move:>8s} {unreal:>14s} {stop:>12s}")

    trades = state["trades"][: args.limit]
    if trades:
        print(f"\n  LAST {len(trades)} CLOSED TRADE(S)")
        print(f"    {'EXIT':12s} {'SYMBOL':12s} {'WHY':5s} {'RET':>8s} "
              f"{'P&L':>14s} {'EQUITY':>15s}")
        for t in trades:
            print(f"    {t['exit_date']:12s} {t['symbol']:12s} {t['exit_reason']:5s} "
                  f"{t['net_return']:+8.2%} {t['pnl']:+14,.2f} {t['equity_after']:15,.2f}")
    else:
        print("\n  No closed trades yet. The account moves when `titan scan` emits a")
        print("  signal and `titan track resolve` grades it — an empty ledger means")
        print("  the gate has refused everything so far, not that tracking is broken.")
    print()
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from titan.models.registry import ModelRegistry
    from titan.monitor.paper import PaperTrackingStore

    cfg = _load_cfg(args)
    registry = ModelRegistry(cfg.model.store_dir)
    store_path = _paper_store_path(cfg)
    tracking = None
    if store_path.exists():
        s = PaperTrackingStore(store_path).summary(_baseline_brier(cfg))
        tracking = {k: s[k] for k in
                    ("n_predictions", "n_resolved", "n_open", "hit_rate", "cusum_alarm")}
    print(json.dumps({
        "version": __version__,
        "provider": cfg.data.provider,
        "universe": [i.symbol for i in cfg.universe.instruments],
        "benchmark": cfg.universe.benchmark,
        "models": [m.to_dict() for m in registry.history()],
        "production": registry.production_version(),
        "paper_tracking": tracking,
        "artifacts_dir": str(Path(cfg.run.artifacts_dir).resolve()),
    }, indent=1, default=str))
    return 0


# --------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="titan",
        description="PROJECT TITAN — quantitative research platform",
    )
    parser.add_argument("--version", action="version", version=f"titan {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_val = sub.add_parser("validate", help="run walk-forward validation and write artifacts")
    p_val.add_argument("--config", help="YAML config path")
    p_val.add_argument("--out", help="artifacts output dir")
    p_val.add_argument("--folds", type=int, help="override cv.n_folds")
    p_val.add_argument("--tuning", type=int, help="override model.tuning_iterations")
    p_val.add_argument("--seed", type=int, help="override run.seed (seed-sensitivity runs)")
    p_val.set_defaults(func=cmd_validate)

    p_scan = sub.add_parser("scan", help="scan the universe with the production model")
    p_scan.add_argument("--config", help="YAML config path (must match the validate run's config)")
    p_scan.add_argument("--out", help="artifacts output dir")
    p_scan.add_argument(
        "--model",
        help=(
            "registry version to scan with (default: the production model). "
            "The config fingerprint is still enforced against whichever model "
            "is chosen — use this to scan with a challenger that validate "
            "produced but the promotion gate did not promote."
        ),
    )
    p_scan.add_argument(
        "--allow-config-mismatch", action="store_true",
        help="scan even if the production model was validated under a different config",
    )
    p_scan.set_defaults(func=cmd_scan)

    p_dash = sub.add_parser("dashboard", help="serve the dashboard")
    p_dash.add_argument("--config", help="YAML config path")
    p_dash.add_argument("--artifacts", help="artifacts dir to serve")
    p_dash.add_argument("--host", default="127.0.0.1")
    p_dash.add_argument("--port", type=int, default=8321)
    p_dash.set_defaults(func=cmd_dashboard)

    p_exp = sub.add_parser(
        "export", help="write a single self-contained dashboard HTML file (no server needed)"
    )
    p_exp.add_argument("--config", help="YAML config path")
    p_exp.add_argument("--artifacts", help="artifacts dir to export (default: config artifacts_dir)")
    p_exp.add_argument("--out", help="output HTML path (default: <artifacts>/titan_dashboard.html)")
    p_exp.set_defaults(func=cmd_export)

    p_track = sub.add_parser(
        "track", help="paper-tracking: grade logged predictions, show live calibration"
    )
    p_track.add_argument(
        "action", choices=["resolve", "status"],
        help="resolve = fetch data and grade elapsed predictions; status = report only",
    )
    p_track.add_argument("--config", help="YAML config path (must match the validate run's config)")
    p_track.add_argument("--out", help="artifacts dir for tracking.json (default: config artifacts_dir)")
    p_track.add_argument(
        "--allow-config-mismatch", action="store_true",
        help="resolve even if the production model was validated under a different config",
    )
    p_track.set_defaults(func=cmd_track)

    p_pre = sub.add_parser(
        "preflight",
        help="QC a universe without running research (minutes, not hours)",
    )
    p_pre.add_argument("--config", help="YAML config path")
    p_pre.set_defaults(func=cmd_preflight)

    p_acct = sub.add_parser(
        "account",
        help="forward paper account: equity, open positions and closed trades",
    )
    p_acct.add_argument("--config", help="YAML config path")
    p_acct.add_argument("--out", help="artifacts output dir")
    p_acct.add_argument("--limit", type=int, default=20, help="closed trades to print")
    p_acct.add_argument(
        "--no-marks", action="store_true",
        help="skip loading prices; report realized P&L only, with no open-book valuation",
    )
    p_acct.set_defaults(func=cmd_account)

    p_info = sub.add_parser("info", help="show platform status")
    p_info.add_argument("--config", help="YAML config path")
    p_info.set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
