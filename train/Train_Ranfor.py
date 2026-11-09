"""Train one Random Forest per label, using only observed training labels."""

import argparse

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from train.common import add_shared_arguments, create_run, evaluate, finish_run, load_splits, save_json


def predict(models, x):
    scores = np.zeros((len(x), len(models)), dtype=np.float32)
    for j, model in enumerate(models):
        positive = np.flatnonzero(model.classes_ == 1)
        if len(positive):
            scores[:, j] = model.predict_proba(x)[:, positive[0]]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20, help="0 for unrestricted trees")
    parser.add_argument("--min-leaf", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    if min(args.trees, args.min_leaf, args.bits) <= 0 or args.max_depth < 0 or args.radius < 0 or args.jobs == 0:
        parser.error("Invalid tree count, dimensions, leaf size, depth, radius or jobs.")
    splits, labels, split_meta = load_splits(args.split, args.radius, args.bits)
    run = create_run("ranfor", args, labels, split_meta)
    x, y, mask = splits["train"]
    models = []
    for j, label in enumerate(labels):
        observed = mask[:, j]
        model = RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth or None,
                                       min_samples_leaf=args.min_leaf, max_features="sqrt",
                                       class_weight="balanced_subsample" if args.balanced else None,
                                       random_state=args.seed, n_jobs=args.jobs)
        model.fit(x[observed], y[observed, j])
        models.append(model)
        print(f"Fitted {j+1}/{len(labels)}: {label}", flush=True)
    joblib.dump({"models": models, "labels": labels, "bits": args.bits, "radius": args.radius},
                run / "model.joblib", compress=3)
    names = ["train", "val"] + (["test"] if args.evaluate_test else [])
    for name in names:
        sx, sy, sm = splits[name]
        result = evaluate(sy, predict(models, sx), sm, labels)
        save_json(run / f"{name}.json", result)
        print(f"{name} macro AP={result['macro_ap']:.4f}", flush=True)
    save_json(run / "summary.json", {"forests": len(models), "trees_per_label": args.trees,
                                     "total_trees": sum(len(m.estimators_) for m in models),
                                     "total_nodes": sum(t.tree_.node_count for m in models for t in m.estimators_),
                                     "test_evaluated": args.evaluate_test})
    finish_run(run)
    print(f"Saved run: {run}")


if __name__ == "__main__":
    main()
