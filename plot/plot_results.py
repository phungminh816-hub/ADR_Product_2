"""Visualization module for ADR_Product2 models and metric evaluation.

Provides command-line plotting tools using matplotlib and seaborn:
- Learning curves for older runs with a saved history.json
- Benchmark model comparison (Side-by-side model comparison)
- Test report comparison (Overall metrics and per-label AP differences)
- Per-label performance ranking (average precision and positive sample counts)
- Sample size vs AP correlation scatter plots

Examples:
    # 1. Chỉ định một thư mục run cụ thể:
    python -m plot.plot_results --run metric/sider/xgboost_res_sider/TEN_RUN --show

    # 2. Chỉ định file đánh giá cụ thể:
    python -m plot.plot_results --file metric/sider/xgboost_res_sider/TEN_RUN/val.json --show --top 20

    # 3. So sánh 2 run bất kỳ do bạn chọn:
    python -m plot.plot_results --compare metric/sider/xgboost_res_sider/RUN1 metric/sider/ranfor_res_sider/RUN2 --show

    # 4. Tự động lấy các bản chạy mới nhất trong metric/ (mặc định):
    python -m plot.plot_results --show
    python -m plot.plot_results --save
"""

import argparse
from pathlib import Path
from textwrap import shorten

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from data import ROOT
from train.runs import find_latest_run, load_json, run_folder

METRIC_DIR = ROOT / "metric"

# Apply professional seaborn styling
sns.set_theme(style="whitegrid", palette="muted")


def finish_figure(fig, path):
    try:
        if path is None:
            plt.show()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300)
            print(f"Saved: {path}")
    finally:
        plt.close(fig)


def plot_single_learning_curve(history_path, title=None, save_path=None):
    """Plot learning curves for a single run given its history.json."""
    history = load_json(history_path)
    if not isinstance(history, list) or not history:
        print(f"Invalid history data in {history_path}")
        return

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_ap = [entry["val_macro_ap"] for entry in history]
    best_idx = int(np.argmax(val_ap))
    best_epoch = epochs[best_idx]
    best_ap = val_ap[best_idx]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, color="#d9534f", linewidth=2.2, label="Train Loss", marker="o", markersize=4)
    ax.plot(epochs, val_ap, color="#0275d8", linewidth=2.2, label="Val Macro AP", marker="s", markersize=4)

    ax.axvline(x=best_epoch, color="#5cb85c", linestyle="--", linewidth=1.5,
               label=f"Peak validation AP (epoch {best_epoch}: {best_ap:.4f})")
    ax.scatter([best_epoch], [best_ap], color="#5cb85c", s=90, zorder=5)

    ax.set_title(title or f"Learning Curves - {history_path.parent.name}", fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("Score / Loss", fontsize=11)
    ax.legend(frameon=True)

    plt.tight_layout()
    finish_figure(fig, save_path)


def plot_comparison_curves(run1, run2, label1, label2, save_path=None):
    """Compare two saved epoch histories."""
    h1_path, h2_path = run1 / "history.json", run2 / "history.json"
    if not h1_path.exists() or not h2_path.exists():
        print(f"Both runs must have history.json to plot line curves ({run1.name}, {run2.name}).")
        return

    h1, h2 = load_json(h1_path), load_json(h2_path)
    s1 = load_json(run1 / "summary.json") if (run1 / "summary.json").exists() else {}
    s2 = load_json(run2 / "summary.json") if (run2 / "summary.json").exists() else {}

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
    plt.subplots_adjust(wspace=0.22)

    configs = [
        (axes[0], h1, s1, label1, "#1f77b4"),
        (axes[1], h2, s2, label2, "#ff7f0e")
    ]

    for ax, history, summary, label, theme_color in configs:
        epochs = [entry["epoch"] for entry in history]
        train_loss = [entry["train_loss"] for entry in history]
        val_ap = [entry["val_macro_ap"] for entry in history]

        best_epoch = summary.get("best_epoch")
        best_idx = epochs.index(best_epoch) if best_epoch in epochs else int(np.argmax(val_ap))
        best_epoch, best_ap = epochs[best_idx], val_ap[best_idx]

        # Plot curves
        ax.plot(epochs, train_loss, color="#d9534f", linewidth=2.2, label="Train Loss", marker="o", markersize=4)
        ax.plot(epochs, val_ap, color=theme_color, linewidth=2.2, label="Val Macro AP", marker="s", markersize=4)

        # Highlight best checkpoint
        ax.axvline(x=best_epoch, color="#2ca02c", linestyle="--", linewidth=1.5,
                   label=f"Selected epoch {best_epoch} (AP {best_ap:.4f})")
        ax.scatter([best_epoch], [best_ap], color="#2ca02c", s=110, edgecolors="black", zorder=5)

        ax.set_title(f"Learning Dynamics: {label} ({len(epochs)} Epochs)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Epoch", fontsize=11, fontweight="bold")
        ax.set_ylabel("Score / Loss", fontsize=11, fontweight="bold")
        ax.set_ylim(0, max(max(train_loss), max(val_ap)) * 1.08)
        ax.legend(frameon=True, loc="center right")

    plt.tight_layout()
    finish_figure(fig, save_path)


def plot_model_comparison(models_data, save_path=None):
    """Plot grouped bar chart comparing Macro AP, ROC-AUC, Micro F1 across models."""
    n_groups = len(models_data)
    fig, axes = plt.subplots(1, n_groups, figsize=(7 * n_groups, 5.5))
    if n_groups == 1:
        axes = [axes]
    plt.subplots_adjust(wspace=0.25)

    metrics = ["macro_ap", "macro_roc_auc", "micro_f1"]
    metric_labels = ["Macro AP", "Macro ROC-AUC", "Micro F1"]

    for idx, (group_title, models) in enumerate(models_data.items()):
        ax = axes[idx]
        x = np.arange(len(metrics))
        n_models = len(models)
        width = 0.8 / max(n_models, 1)

        colors = ["#4a90e2", "#50e3c2", "#f5a623", "#e94e77"]
        for m_idx, (m_name, m_data) in enumerate(models.items()):
            scores = [m_data.get(metric) for metric in metrics]
            offset = (m_idx - (n_models - 1) / 2) * width
            ax.bar(x + offset, [score if score is not None else np.nan for score in scores],
                   width, label=m_name,
                   color=colors[m_idx % len(colors)], edgecolor="black", alpha=0.9)
            for position, score in zip(x + offset, scores):
                height = score if score is not None else 0
                label = f"{score:.3f}" if score is not None else "N/A"
                ax.text(position, height + .015, label, ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Comparison - {group_title}", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
        ax.legend(frameon=True)

    plt.tight_layout()
    finish_figure(fig, save_path)


def plot_test_report(report, save_dir=None, top=8):
    """Plot overall test metrics and the largest AP differences by ADR label."""
    models = report["test"]
    forest = models["Random Forest"]
    booster_name, = (name for name in models if name != "Random Forest")
    booster = models[booster_name]
    dataset = {"sider": "SIDER", "onsides": "OnSIDES"}[report["dataset"]]
    title = f"{dataset} test (n={booster['samples']})"
    differences = []
    for label, metrics in booster["per_label"].items():
        reference = forest["per_label"][label]["ap"]
        if metrics["ap"] is not None and reference is not None:
            name = label.partition(" ")[2] if label.startswith("MedDRA:") and " " in label else label
            caption = f"{shorten(name, width=48)}  (n+={metrics['positive']})"
            differences.append((metrics["ap"] - reference, caption))
    delta_ap = report["xgboost_minus_random_forest"]["macro_ap"]
    delta_text = f"{delta_ap:+.3f}" if delta_ap is not None else "N/A"
    overall_title = f"{title}\nΔ Macro AP (XGBoost − Random Forest) {delta_text}"
    overall_path = save_dir / "comparison.png" if save_dir else None
    plot_model_comparison({overall_title: models}, overall_path)

    if not differences:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.set_axis_off()
        ax.text(.5, .5, "No ADR label has a positive test example.", ha="center", va="center")
        ax.set_title(title)
        finish_figure(fig, save_dir / "per_label_ap_delta.png" if save_dir else None)
        return

    differences.sort()
    selected = ([row for row in differences if row[0] < 0][:top]
                + [row for row in differences if row[0] > 0][-top:])
    selected = selected or differences[:top]
    values, labels = zip(*selected)
    positions = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(10, max(3, 2 + .37 * len(labels))), layout="constrained")
    ax.barh(positions, values, color=["#4a90e2" if value > 0 else "#50b8a3" for value in values])
    ax.set_yticks(positions, labels)
    ax.axvline(0, color="#3f3f46", linewidth=1)
    limit = max(.01, max(map(abs, values)))
    ax.set_xlim(-limit * 1.24, limit * 1.24)
    for position, value in zip(positions, values):
        offset = limit * .025 if value >= 0 else -limit * .025
        ax.text(value + offset, position, f"{value:+.3f}", va="center",
                ha="left" if value >= 0 else "right", fontsize=9)
    ax.set_title(f"{title} | AP difference by ADR label", fontweight="bold")
    ax.set_xlabel("XGBoost AP − Random Forest AP")
    ax.grid(False, axis="y")
    finish_figure(fig, save_dir / "per_label_ap_delta.png" if save_dir else None)


def plot_per_label_ranking(val_data, title, save_path=None, top_n=15):
    """Plot horizontal bar chart of top performing ADR categories."""
    per_label = val_data.get("per_label", {})
    if not per_label:
        return

    valid_items = [(name, m) for name, m in per_label.items() if m.get("ap") is not None]
    valid_items.sort(key=lambda item: item[1]["ap"], reverse=True)
    top_items = valid_items[:top_n][::-1]
    labels, ap_scores = [], []
    for name, metrics in top_items:
        term = name.partition(" ")[2] if name.startswith("MedDRA:") and " " in name else name
        caption = shorten(term, width=48)
        if metrics.get("positive") is not None:
            caption += f" (n+={metrics['positive']})"
        labels.append(caption)
        ap_scores.append(metrics["ap"])

    fig, ax = plt.subplots(figsize=(10, max(5.5, top_n * 0.38)))
    positions = np.arange(len(labels))
    bars = ax.barh(positions, ap_scores, color="#5c6bc0", edgecolor="#283593", height=0.65)
    ax.set_yticks(positions, labels)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height()/2., f"{w:.3f}", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(0, max(ap_scores) * 1.18 if ap_scores else 1.0)
    ax.set_xlabel("Average Precision (AP)", fontsize=11, fontweight="bold")
    ax.set_title(f"Top {len(top_items)} ADR labels by AP - {title}", fontsize=12, fontweight="bold", pad=12)

    plt.tight_layout()
    finish_figure(fig, save_path)


def plot_sample_vs_ap(val_data, title, save_path=None, partition="Validation"):
    """Plot positive sample counts and AP for the selected partition."""
    per_label = val_data.get("per_label", {})
    if not per_label:
        return

    positives, aps = [], []
    for v in per_label.values():
        if v.get("ap") is not None and v.get("positive") is not None:
            positives.append(v["positive"])
            aps.append(v["ap"])

    if not positives:
        return

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    scatter = ax.scatter(positives, aps, c=aps, cmap="viridis", s=70, edgecolors="black", alpha=0.85)

    if min(positives) != max(positives):
        z = np.polyfit(positives, aps, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(positives), max(positives), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, label=f"Trendline (slope={z[0]:.4f})")
        ax.legend(frameon=True)

    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Average Precision (AP)", fontsize=10)

    ax.set_title(f"Positive samples vs AP ({title})", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel(f"Positive samples ({partition})", fontsize=11)
    ax.set_ylabel(f"Average precision ({partition})", fontsize=11)

    plt.tight_layout()
    finish_figure(fig, save_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--run", type=Path, help="Path to a specific run folder")
    source.add_argument("--file", type=Path, help="Metrics JSON (train/val/test) or a legacy history.json")
    source.add_argument("--test-report", type=Path, help="Render the figures from a saved test comparison.json")
    parser.add_argument("--metric-dir", type=Path, default=METRIC_DIR, help="Directory to scan for validation runs")
    source.add_argument("--compare", nargs="+", type=Path, metavar="RUN",
                        help="Compare two or more run folders side-by-side")
    parser.add_argument("--top", type=int, help="Number of labels in a ranking or per direction of test AP difference")
    parser.add_argument("--line", action="store_true", help="Plot comparison as line chart over epochs (Loss & Val Macro AP) instead of bar chart")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--save", action="store_true", help="Save PNG files (default)")
    mode.add_argument("--show", action="store_true", help="Show plots interactively in GUI windows")
    parser.add_argument("--output", type=Path, help="Output directory (default: next to the test report or metric/figures)")
    parser.add_argument("--name", type=str, help="Custom filename for the saved plot image (e.g. sider_curves.png)")
    parser.add_argument("--chart", choices=["all", "learning_curves", "comparison", "ranking", "scatter"],
                        default="all", help="Select which chart to generate (default: all)")
    args = parser.parse_args()
    if args.top is not None and args.top <= 0:
        parser.error("--top must be positive.")
    if args.line and (not args.compare or len(args.compare) != 2):
        parser.error("--line requires --compare with two runs containing history.json.")
    if args.name and not args.compare:
        parser.error("--name applies to --compare.")

    # Determine saving target
    default_output = args.test_report.parent if args.test_report else args.metric_dir / "figures"
    save_dir = (args.output or default_output) if (args.save or not args.show) else None
    if args.test_report:
        if args.chart not in ("all", "comparison"):
            parser.error("--test-report renders test comparisons.")
        try:
            plot_test_report(load_json(args.test_report), save_dir, top=args.top or 8)
        except (OSError, TypeError, ValueError, KeyError) as error:
            parser.error(str(error))
        return
    top = args.top or 15

    # CASE 1: User explicitly passed a single JSON file
    if args.file:
        file_path = args.file.resolve()
        if not file_path.exists():
            parser.error(f"File not found: {file_path}")
        data = load_json(file_path)

        # Detect file type
        if "history" in file_path.name or isinstance(data, list):
            if args.chart not in ("all", "learning_curves"):
                parser.error("A training history supports --chart learning_curves.")
            out = (save_dir / f"{file_path.stem}_curve.png") if save_dir else None
            plot_single_learning_curve(file_path, title=f"History from {file_path.parent.name}", save_path=out)
        elif "per_label" in data:
            if args.chart not in ("all", "ranking", "scatter"):
                parser.error("Per-label metrics support ranking and scatter.")
            partition = {"val": "Validation", "train": "Training", "test": "Test"}.get(file_path.stem, file_path.stem)
            title = f"{file_path.parent.name} ({partition})"
            stem = file_path.parent.name if file_path.stem == "val" else f"{file_path.parent.name}_{file_path.stem}"
            if args.chart in ("all", "ranking"):
                out = (save_dir / f"{stem}_ranking.png") if save_dir else None
                plot_per_label_ranking(data, title, save_path=out, top_n=top)
            if args.chart in ("all", "scatter"):
                out = (save_dir / f"{stem}_scatter.png") if save_dir else None
                plot_sample_vs_ap(data, title, save_path=out, partition=partition)
        else:
            parser.error(f"Expected per-label metrics or a training history in {file_path}")
        return

    # CASE 2: User explicitly passed runs to compare
    if args.compare:
        if args.chart not in ("all", "learning_curves" if args.line else "comparison"):
            parser.error("Use --compare for metric comparisons, or add --line for saved histories.")
        runs = [r.resolve() for r in args.compare]
        if len(set(runs)) != len(runs) or len(runs) < 2:
            parser.error("--compare requires at least two distinct runs.")
        for r in runs:
            if not r.exists() or not (r / "val.json").exists():
                parser.error(f"Run folder must exist and contain val.json: {r}")

        def get_label(r):
            if not (r / "metadata.json").is_file() or not (r / "split.json").is_file():
                return r.name
            meta = load_json(r / "metadata.json")
            split = load_json(r / "split.json")
            model = {"xgboost": "XGBoost", "ranfor": "Random Forest"}.get(meta.get("model"), meta.get("model", r.name))
            dataset = {"sider": "SIDER", "onsides": "OnSIDES"}.get(meta.get("dataset"), "")
            return f"{dataset} {model} - {split['strategy']} (seed {split['seed']})".strip()

        if args.line and len(runs) == 2:
            if not all((run / "history.json").is_file() for run in runs):
                parser.error("--line requires history.json in both runs. Current tree trainers save final metrics.")
            label1, label2 = get_label(runs[0]), get_label(runs[1])
            fname = args.name or "comparison_curves.png"
            out = (save_dir / fname) if save_dir else None
            plot_comparison_curves(runs[0], runs[1], label1, label2, save_path=out)
        else:
            group_dict = {}
            for index, r in enumerate(runs, start=1):
                lbl = get_label(r)
                if lbl in group_dict:
                    lbl = f"{lbl} (run {index})"
                group_dict[lbl] = load_json(r / "val.json")
            models_data = {"Benchmark Comparison": group_dict}
            fname = args.name or "comparison_benchmark.png"
            out = (save_dir / fname) if save_dir else None
            plot_model_comparison(models_data, save_path=out)
        return

    # CASE 3: User explicitly passed a single run folder
    if args.run:
        run_path = args.run.resolve()
        if not run_path.exists():
            parser.error(f"Run folder not found: {run_path}")
        if args.chart == "comparison":
            parser.error("Use --compare with two or more runs.")
        if args.chart == "learning_curves" and not (run_path / "history.json").is_file():
            parser.error("This run has no history.json. Use --chart ranking or scatter for tree models.")

        if (run_path / "history.json").exists() and args.chart in ("all", "learning_curves"):
            out = (save_dir / f"{run_path.name}_learning_curve.png") if save_dir else None
            plot_single_learning_curve(run_path / "history.json", title=f"Training Dynamics - {run_path.name}", save_path=out)

        val_file = run_path / "val.json"
        if not val_file.is_file() and (args.chart in ("ranking", "scatter") or not (run_path / "history.json").is_file()):
            parser.error(f"Run is missing validation metrics: {val_file}")
        if val_file.exists():
            val_data = load_json(val_file)
            if args.chart in ("all", "ranking"):
                out = (save_dir / f"{run_path.name}_ranking.png") if save_dir else None
                plot_per_label_ranking(val_data, run_path.name, save_path=out, top_n=top)
            if args.chart in ("all", "scatter"):
                out = (save_dir / f"{run_path.name}_scatter.png") if save_dir else None
                plot_sample_vs_ap(val_data, run_path.name, save_path=out)
        return

    # CASE 4: Default fallback - auto scan latest runs in metric/
    sider_xgb = find_latest_run(run_folder(args.metric_dir, "sider", "xgboost"))
    sider_rf = find_latest_run(run_folder(args.metric_dir, "sider", "ranfor"))
    onside_xgb = find_latest_run(run_folder(args.metric_dir, "onsides", "xgboost"))
    onside_rf = find_latest_run(run_folder(args.metric_dir, "onsides", "ranfor"))

    if args.chart == "learning_curves":
        parser.error("Select a run or file with a saved training history.")
    if args.chart in ("all", "ranking", "scatter") and not (sider_xgb or onside_xgb):
        parser.error("No completed XGBoost run for automatic plots. Use --run to inspect a Random Forest run.")

    if args.chart in ("all", "comparison"):
        models_data = {}
        for dataset, booster, forest in (("SIDER", sider_xgb, sider_rf),
                                         ("OnSIDES", onside_xgb, onside_rf)):
            if booster is None or forest is None:
                continue
            if load_json(booster / "split.json")["files"] != load_json(forest / "split.json")["files"]:
                continue
            loss = load_json(booster / "summary.json")["loss"].upper()
            models_data[f"{dataset} validation"] = {
                f"XGBoost ({loss})": load_json(booster / "val.json"),
                "Random Forest": load_json(forest / "val.json"),
            }
        if models_data:
            out = (save_dir / "model_comparison.png") if save_dir else None
            plot_model_comparison(models_data, out)
        elif args.chart == "comparison":
            parser.error("No completed model pair with matching validation splits.")

    if args.chart in ("all", "ranking"):
        if sider_xgb and (sider_xgb / "val.json").exists():
            out = (save_dir / "per_label_sider.png") if save_dir else None
            plot_per_label_ranking(load_json(sider_xgb / "val.json"), "SIDER (XGBoost)", out, top_n=top)
        if onside_xgb and (onside_xgb / "val.json").exists():
            out = (save_dir / "per_label_onside.png") if save_dir else None
            plot_per_label_ranking(load_json(onside_xgb / "val.json"), "OnSIDES (XGBoost)", out, top_n=top)

    if args.chart in ("all", "scatter"):
        if sider_xgb and (sider_xgb / "val.json").exists():
            out = (save_dir / "sample_vs_ap_sider.png") if save_dir else None
            plot_sample_vs_ap(load_json(sider_xgb / "val.json"), "SIDER", out)
        if onside_xgb and (onside_xgb / "val.json").exists():
            out = (save_dir / "sample_vs_ap_scatter.png") if save_dir else None
            plot_sample_vs_ap(load_json(onside_xgb / "val.json"), "OnSIDES", out)


if __name__ == "__main__":
    main()
