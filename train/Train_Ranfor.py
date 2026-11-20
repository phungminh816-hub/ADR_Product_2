"""Train one Random Forest per label, using only observed training labels."""

import argparse

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
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
    parser.add_argument("--max-features", type=float, help="Fraction of features per split (default: sqrt)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()
    if min(args.trees, args.min_leaf, args.bits) <= 0 or args.max_depth < 0 or args.radius < 0 or args.jobs == 0:
        parser.error("Invalid tree count, dimensions, leaf size, depth, radius or jobs.")
    if args.max_features is not None and not 0 < args.max_features <= 1:
        parser.error("Max features must be in (0, 1].")
    names = ("train", "val", "test") if args.evaluate_test else ("train", "val")
    splits, labels, split_meta = load_splits(args.split, args.radius, args.bits, parts=names)
    run = create_run("ranfor", args, labels, split_meta)
    x, y, mask = splits["train"]
    models = []
    max_features = "sqrt" if args.max_features is None else args.max_features
    for j in range(len(labels)):
        observed = mask[:, j]
        target = y[observed, j]
        if np.all(target == target[0]):
            model = DummyClassifier(strategy="constant", constant=int(target[0]))
        else:
            model = RandomForestClassifier(n_estimators=args.trees, max_depth=args.max_depth or None,
                                           min_samples_leaf=args.min_leaf, max_features=max_features,
                                           class_weight="balanced_subsample" if args.balanced else None,
                                           random_state=args.seed, n_jobs=args.jobs)
        model.fit(x[observed], target)
        models.append(model)
    joblib.dump({"models": models, "labels": labels, "bits": args.bits, "radius": args.radius},
                run / "model.joblib", compress=3)
    for name in names:
        sx, sy, sm = splits[name]
        save_json(run / f"{name}.json", evaluate(sy, predict(models, sx), sm, labels))
    forests = [model for model in models if isinstance(model, RandomForestClassifier)]
    save_json(run / "summary.json", {"forests": len(forests), "constant_labels": len(models) - len(forests),
                                     "trees_per_label": args.trees,
                                     "total_trees": sum(len(m.estimators_) for m in forests),
                                     "total_nodes": sum(t.tree_.node_count for m in forests for t in m.estimators_),
                                     "test_evaluated": args.evaluate_test})
    finish_run(run)
    print(f"Saved run: {run}")


if __name__ == "__main__":
    main()
