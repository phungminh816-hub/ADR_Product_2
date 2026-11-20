"""Load a saved XGBoost or Random Forest run and predict ADR scores for one SMILES."""

import argparse
from functools import partial
from pathlib import Path

import joblib
import numpy as np

from train.common import fingerprints, load_json
from train.Train_Ranfor import predict as forest_predict
from train.Train_XGBoost import load_models, predict as booster_predict


def load_predictor(run):
    """Return the label order, fingerprint settings and scoring function for a run."""
    metadata = load_json(run / "metadata.json") if (run / "metadata.json").is_file() else None
    if metadata is not None and metadata.get("status") != "complete":
        raise ValueError(f"Training run is incomplete: {run}")
    booster = (run / "boosters.json").is_file()
    forest = (run / "model.joblib").is_file()
    if not (booster or forest):
        raise FileNotFoundError(f"No XGBoost or Random Forest checkpoint found in {run}")
    family = "xgboost" if booster else "ranfor"
    if (booster and forest) or (metadata is not None and metadata.get("model") != family):
        raise ValueError(f"Inconsistent model type in {run}")
    if booster:
        config = load_json(run / "config.json")
        labels = load_json(run / "labels.json")
        models = load_models(run)
        predict = booster_predict
    else:
        config = joblib.load(run / "model.joblib")
        labels, models = config["labels"], config["models"]
        if (run / "labels.json").is_file() and labels != load_json(run / "labels.json"):
            raise ValueError(f"Checkpoint label order does not match labels.json: {run}")
        predict = forest_predict
    if not labels or len(models) != len(labels) or len(set(labels)) != len(labels):
        raise ValueError(f"Expected one model per unique label: {run}")
    return labels, config["radius"], config["bits"], partial(predict, models)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    labels, radius, bits, score = load_predictor(args.run)
    scores = score(fingerprints([args.smiles], radius, bits))[0]
    for j in np.argsort(-scores)[:5]:
        print(f"{labels[j]}: {scores[j]:.4f}")


if __name__ == "__main__":
    main()
