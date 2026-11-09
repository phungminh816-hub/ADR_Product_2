"""Load a saved MLP or Random Forest run and predict ADR scores for one SMILES."""

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch

from train.common import fingerprints
from train.Train_MLP import MLP, predict as mlp_predict
from train.Train_Ranfor import predict as forest_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    if (args.run / "model.pt").exists():
        saved = torch.load(args.run / "model.pt", map_location="cpu", weights_only=True)
        model = MLP(saved["bits"], saved["hidden"], saved["dropout"], len(saved["labels"]))
        model.load_state_dict(saved["state_dict"])
        x = fingerprints([args.smiles], saved["radius"], saved["bits"])
        scores = mlp_predict(model, x, "cpu")[0]
    else:
        saved = joblib.load(args.run / "model.joblib")
        x = fingerprints([args.smiles], saved["radius"], saved["bits"])
        scores = forest_predict(saved["models"], x)[0]
    for j in np.argsort(-scores)[:5]:
        print(f"{saved['labels'][j]}: {scores[j]:.4f}")


if __name__ == "__main__":
    main()
