"""Train one XGBoost booster per ADR label using asymmetric loss."""

import argparse
import math
from functools import partial

import numpy as np
import xgboost as xgb
from scipy.special import expit

from train.asymmetric_loss import asl_objective
from train.common import (
    add_shared_arguments, create_run, evaluate, finish_run, load_json, load_splits,
    require_scaffold, save_json,
)


def predict(models, x):
    scores = np.empty((len(x), len(models)), dtype=np.float32)
    for j, model in enumerate(models):
        if isinstance(model, xgb.Booster):
            scores[:, j] = expit(model.inplace_predict(x, predict_type="margin"))
        else:
            scores[:, j] = model
    return scores


def load_models(run):
    constants = load_json(run / "boosters.json")
    return [xgb.Booster(params={"nthread": 2}, model_file=run / "boosters" / f"{j:03d}.ubj")
            if value is None else float(value) for j, value in enumerate(constants)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--lr", type=float, default=.1)
    parser.add_argument("--min-child-weight", type=float, default=1.)
    parser.add_argument("--colsample-bytree", type=float, default=1.)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--gamma-pos", type=float, default=1.)
    parser.add_argument("--gamma-neg", type=float, default=4.)
    parser.add_argument("--clip", type=float, default=.05)
    args = parser.parse_args()
    if (min(args.rounds, args.max_depth, args.jobs, args.bits) <= 0 or args.radius < 0
            or not 0 < args.lr <= 1):
        parser.error("Invalid rounds, depth, jobs, fingerprint dimensions or learning rate.")
    if not math.isfinite(args.min_child_weight) or args.min_child_weight < 0:
        parser.error("Min child weight must be finite and nonnegative.")
    if not 0 < args.colsample_bytree <= 1:
        parser.error("Column sample must be in (0, 1].")
    if not all(math.isfinite(gamma) and gamma >= 0 for gamma in (args.gamma_pos, args.gamma_neg)):
        parser.error("ASL gamma values must be finite and nonnegative.")
    if not 0 <= args.clip < 1:
        parser.error("ASL clip must be in [0, 1).")

    names = ("train", "val", "test") if args.evaluate_test else ("train", "val")
    splits, labels, split_meta = load_splits(args.split, args.radius, args.bits, parts=names)
    require_scaffold(split_meta)
    run = create_run("xgboost", args, labels, split_meta)
    (run / "boosters").mkdir()
    x, y, mask = splits["train"]
    objective = partial(asl_objective, gamma_pos=args.gamma_pos, gamma_neg=args.gamma_neg, clip=args.clip)
    params = {
        "objective": "binary:logistic", "base_score": .5, "tree_method": "hist",
        "max_depth": args.max_depth, "eta": args.lr, "max_delta_step": 1,
        "min_child_weight": args.min_child_weight, "colsample_bytree": args.colsample_bytree,
        "nthread": args.jobs, "seed": args.seed,
    }
    models, constants = [], []
    full_matrix = xgb.DMatrix(x, nthread=args.jobs)
    for j in range(len(labels)):
        observed = mask[:, j]
        target = y[observed, j]
        if np.all(target == target[0]):
            constant = float(target[0])
            models.append(constant)
            constants.append(constant)
            continue
        matrix = full_matrix if observed.all() else xgb.DMatrix(x[observed], nthread=args.jobs)
        matrix.set_label(target)
        booster = xgb.train(params, matrix, num_boost_round=args.rounds, obj=objective)
        booster.save_model(run / "boosters" / f"{j:03d}.ubj")
        models.append(booster)
        constants.append(None)

    save_json(run / "boosters.json", constants)
    for name in names:
        sx, sy, sm = splits[name]
        save_json(run / f"{name}.json", evaluate(sy, predict(models, sx), sm, labels))
    save_json(run / "summary.json", {
        "loss": "bce" if args.gamma_pos == args.gamma_neg == args.clip == 0 else "asl",
        "boosters": constants.count(None),
        "constant_labels": len(labels) - constants.count(None),
        "rounds_per_booster": args.rounds, "test_evaluated": args.evaluate_test,
    })
    finish_run(run)
    print(f"Saved run: {run}")


if __name__ == "__main__":
    main()
