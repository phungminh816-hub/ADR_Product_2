"""Evaluate saved XGBoost and Random Forest models on the same scaffold test set."""

import argparse
from pathlib import Path

import matplotlib

from data import DATASET_FOLDERS, ROOT
from train.common import dataset_from_labels, evaluate, load_splits, require_scaffold, save_json
from train.predict import load_predictor
from train.runs import find_latest_run, load_json, run_folder


def latest_run(folder):
    run = find_latest_run(folder)
    if run is None:
        raise ValueError(f"No completed run in {folder}")
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_FOLDERS, default="sider")
    parser.add_argument("--metric-dir", type=Path, default=ROOT / "metric", help="Directory containing runs")
    parser.add_argument("--xgboost-run", type=Path, help="XGBoost run directory (default: latest completed)")
    parser.add_argument("--rf-run", type=Path, help="Random Forest run directory (default: latest completed)")
    parser.add_argument("--output", type=Path, help="Output directory for the test comparison")
    args = parser.parse_args()

    dataset_root = args.metric_dir / DATASET_FOLDERS[args.dataset]
    specs = (
        ("XGBoost", "xgboost", args.xgboost_run),
        ("Random Forest", "ranfor", args.rf_run),
    )
    results, runs = {}, {}
    reference_files = None
    try:
        for name, family, selected in specs:
            run = (selected or latest_run(run_folder(args.metric_dir, args.dataset, family))).resolve()
            metadata = load_json(run / "metadata.json")
            if metadata["model"] != family or metadata["status"] != "complete":
                raise ValueError(f"Expected a completed {family} run: {run}")
            snapshot = run / "split_snapshot"
            labels, radius, bits, score = load_predictor(run)
            if metadata.get("dataset", dataset_from_labels(labels)) != args.dataset:
                raise ValueError(f"Checkpoint dataset does not match --dataset {args.dataset}: {run}")
            splits, snapshot_labels, meta = load_splits(snapshot, radius, bits, parts=("test",))
            require_scaffold(meta)
            if reference_files is not None and meta["files"] != reference_files:
                raise ValueError("The runs must use identical split snapshots and label order.")
            reference_files = meta["files"]
            if labels != snapshot_labels:
                raise ValueError(f"Checkpoint labels do not match the split: {run}")
            if family == "xgboost":
                loss = load_json(run / "summary.json")["loss"].upper()
                name = f"{name} ({loss})"
            x, y, mask = splits["test"]
            results[name] = evaluate(y, score(x), mask, labels)
            runs[name] = str(run)
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    xgb, rf = results.values()
    delta = {}
    for metric in ("macro_ap", "macro_roc_auc", "micro_f1"):
        delta[metric] = xgb[metric] - rf[metric] if xgb[metric] is not None and rf[metric] is not None else None
    output = args.output or (dataset_root / "test_comparison")

    matplotlib.use("Agg")
    from plot.plot_results import plot_test_report

    report = {
        "dataset": args.dataset, "split_files": reference_files, "runs": runs,
        "test": results, "xgboost_minus_random_forest": delta,
    }
    plot_test_report(report, output)
    save_json(output / "comparison.json", report)
    print(f"Saved: {output / 'comparison.json'}")


if __name__ == "__main__":
    main()
