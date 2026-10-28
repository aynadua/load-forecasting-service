"""Command line entry point: ``loadfc <stage>``.

    loadfc data       # download and clean the PJM East series
    loadfc backtest   # rolling-origin evaluation against the baselines
    loadfc train      # fit the final model on all data and export artefacts
    loadfc serve      # run the FastAPI service
    loadfc report     # collect metrics and figures into reports/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from loadfc import backtest as backtest_module
from loadfc import data as data_module
from loadfc import model as model_module
from loadfc import plots
from loadfc.drift import DriftMonitor
from loadfc.features import FeatureSpec, build_features

RAW = Path("data/PJME_hourly.csv")
REPORTS = Path("reports")
MODELS = Path("models")


def _series(args: argparse.Namespace):
    if not RAW.exists():
        raise SystemExit(f"missing {RAW}; run `loadfc data` first")
    return data_module.load_series(RAW)


def cmd_data(args: argparse.Namespace) -> None:
    path = data_module.download(RAW)
    series, report = data_module.load_series(path)
    print(report.describe())
    print(json.dumps(data_module.summarise(series), indent=2))


def cmd_backtest(args: argparse.Namespace) -> None:
    series, _ = _series(args)
    spec = FeatureSpec(horizon=args.horizon)
    result = backtest_module.run(
        series,
        spec,
        n_folds=args.folds,
        test_hours=args.test_hours,
        min_train_hours=args.min_train_hours,
        seed=args.seed,
    )

    record = backtest_module.to_record(result, {"horizon_hours": args.horizon})
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "backtest.json").write_text(json.dumps(record, indent=2))

    result.actuals.to_frame("actual").to_parquet(REPORTS / "actuals.parquet")
    for name, prediction in result.predictions.items():
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        prediction.to_frame("prediction").to_parquet(REPORTS / f"pred_{slug}.parquet")

    print()
    for name, scores in result.summary().items():
        print(
            f"{name:24s} MAE {scores['mae']:7.1f} MW   MAPE {scores['mape']:5.2f}% "
            f"(+/- {scores['mape_std']:.2f})   MASE {scores['mase']:.3f}"
        )


def cmd_train(args: argparse.Namespace) -> None:
    series, _ = _series(args)
    spec = FeatureSpec(horizon=args.horizon)
    matrix = build_features(series, spec)

    bundle = model_module.fit(matrix, seed=args.seed)
    bundle.save(MODELS / "model.joblib")

    monitor = DriftMonitor.from_training(matrix.x)
    MODELS.mkdir(parents=True, exist_ok=True)
    monitor.reference.to_parquet(MODELS / "reference.parquet")

    ranking = model_module.importances(bundle, matrix)
    ranking.to_frame().to_csv(REPORTS / "importance.csv")

    print(f"trained on {bundle.trained_on['n_rows']} rows "
          f"({bundle.trained_on['start']} to {bundle.trained_on['end']})")
    print(f"boosting iterations: {bundle.trained_on['n_iterations']}")
    print("top features:")
    for name, value in ranking.head(8).items():
        print(f"  {name:18s} {value:7.1f} MW")


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("loadfc.service:app", host=args.host, port=args.port, reload=False)


def cmd_report(args: argparse.Namespace) -> None:
    import pandas as pd

    path = REPORTS / "backtest.json"
    if not path.exists():
        raise SystemExit("missing reports/backtest.json; run `loadfc backtest` first")
    record = json.loads(path.read_text())

    series, clean_report = _series(args)
    plots.load_profile(series, REPORTS / "load_profile.png")
    plots.fold_scores(record, REPORTS / "fold_mape.png", metric="mape")

    actuals = pd.read_parquet(REPORTS / "actuals.parquet")["actual"]
    predictions: dict[str, pd.Series] = {}
    for name in record["per_fold"]:
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        candidate = REPORTS / f"pred_{slug}.parquet"
        if candidate.exists():
            predictions[name] = pd.read_parquet(candidate)["prediction"]

    if predictions:
        plots.forecast_window(actuals, predictions, REPORTS / "forecast_window.png")
        plots.error_by_hour(actuals, predictions, REPORTS / "error_by_hour.png")

    importance_path = REPORTS / "importance.csv"
    if importance_path.exists():
        ranking = pd.read_csv(importance_path, index_col=0).iloc[:, 0]
        plots.importance(ranking, REPORTS / "importance.png")

    summary = record["summary"]
    lines = [
        "# Results",
        "",
        f"Data: {clean_report.describe()}",
        "",
        f"Forecast horizon: {record.get('horizon_hours', 24)} hours. "
        f"{len(record['folds'])} expanding-window folds, each tested on one unseen year.",
        "",
        "| Model | MAE (MW) | MAPE | MASE | Peak-hour MAE | MAPE spread |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, scores in summary.items():
        lines.append(
            f"| {name} | {scores['mae']:.0f} | {scores['mape']:.2f}% | {scores['mase']:.3f} | "
            f"{scores['peak_hour_mae']:.0f} | ±{scores['mape_std']:.2f} |"
        )
    lines += ["", "![fold mape](fold_mape.png)", "", "![forecast](forecast_window.png)", ""]

    out = REPORTS / "RESULTS.md"
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loadfc", description=__doc__)
    parser.add_argument("--horizon", type=int, default=24, help="forecast horizon in hours")
    parser.add_argument("--seed", type=int, default=7)
    sub = parser.add_subparsers(dest="stage", required=True)

    sub.add_parser("data", help="download and clean").set_defaults(handler=cmd_data)
    sub.add_parser("train", help="fit the final model").set_defaults(handler=cmd_train)
    sub.add_parser("report", help="collect metrics and figures").set_defaults(handler=cmd_report)

    backtest_parser = sub.add_parser("backtest", help="rolling-origin evaluation")
    backtest_parser.add_argument("--folds", type=int, default=6)
    backtest_parser.add_argument("--test-hours", type=int, default=8760)
    backtest_parser.add_argument(
        "--min-train-hours", type=int, default=26280,
        help="training history required before the first fold (default: 3 years)",
    )
    backtest_parser.set_defaults(handler=cmd_backtest)

    serve_parser = sub.add_parser("serve", help="run the inference service")
    serve_parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.set_defaults(handler=cmd_serve)

    args = parser.parse_args(argv)
    args.handler(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
