"""Shared feature generation, split validation, metrics and run metadata."""

import hashlib
import importlib.metadata
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[1]


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def fingerprints(smiles, radius=2, bits=2048):
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=bits, includeChirality=True)
    result = []
    for text in smiles:
        molecule = Chem.MolFromSmiles(text)
        if molecule is None or molecule.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES: {text}")
        result.append(generator.GetFingerprintAsNumPy(molecule))
    return np.asarray(result, dtype=np.float32)


def load_splits(folder, radius=2, bits=2048):
    folder = folder.resolve()
    meta = json.loads((folder / "split.json").read_text(encoding="utf-8"))
    for filename in ("train.csv", "val.csv", "test.csv", "labels.json"):
        digest = hashlib.sha256((folder / filename).read_bytes()).hexdigest()
        if digest != meta["files"][filename]:
            raise ValueError(f"Split checksum mismatch: {filename}")
    labels = json.loads((folder / "labels.json").read_text(encoding="utf-8"))
    splits, seen = {}, set()
    for name in ("train", "val", "test"):
        frame = pd.read_csv(folder / f"{name}.csv")
        if frame.drug_id.duplicated().any() or seen & set(frame.drug_id):
            raise ValueError("Duplicate or overlapping drug IDs.")
        seen.update(frame.drug_id)
        y = frame[labels].to_numpy(dtype=np.float32)
        mask = np.isfinite(y)
        if not np.isin(y[mask], [0, 1]).all() or np.isinf(y).any():
            raise ValueError("Observed labels must be binary.")
        splits[name] = (fingerprints(frame.smiles, radius, bits), np.nan_to_num(y), mask)
    if np.any(splits["train"][2].sum(axis=0) == 0):
        raise ValueError("At least one label has no observed training samples.")
    return splits, labels, meta


def evaluate(y, probability, mask, labels, threshold=.5):
    if probability.shape != y.shape or not np.isfinite(probability).all():
        raise ValueError("Invalid predictions.")
    if np.any((probability < 0) | (probability > 1)):
        raise ValueError("Prediction scores must be in [0, 1].")
    per_label, aps, rocs = {}, [], []
    for j, label in enumerate(labels):
        observed = mask[:, j]
        truth, score = y[observed, j], probability[observed, j]
        positive = int(truth.sum())
        negative = len(truth) - positive
        ap = float(average_precision_score(truth, score)) if positive else None
        roc = float(roc_auc_score(truth, score)) if positive and negative else None
        if ap is not None:
            aps.append(ap)
        if roc is not None:
            rocs.append(roc)
        per_label[label] = {"positive": positive, "negative": negative,
                            "missing": int((~observed).sum()), "ap": ap, "roc_auc": roc}
    if not aps:
        raise ValueError("No positive labels available for AP evaluation.")
    predicted = probability[mask] >= threshold
    truth = y[mask]
    return {"samples": len(y), "macro_ap": float(np.mean(aps)),
            "macro_roc_auc": float(np.mean(rocs)) if rocs else None,
            "valid_ap_labels": len(aps), "valid_roc_labels": len(rocs),
            "micro_precision": float(precision_score(truth, predicted, zero_division=0)),
            "micro_recall": float(recall_score(truth, predicted, zero_division=0)),
            "micro_f1": float(f1_score(truth, predicted, zero_division=0)),
            "threshold": threshold, "per_label": per_label}


def create_run(model_type, args, labels, split_meta):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run = ROOT / "metric" / f"{model_type}_{stamp}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    save_json(run / "config.json", config)
    save_json(run / "labels.json", labels)
    save_json(run / "split.json", split_meta)
    versions = {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "rdkit", "torch", "joblib")}
    save_json(run / "metadata.json", {"model": model_type, "created_at_utc": stamp,
                                      "versions": versions, "status": "running"})
    # Preserve the exact data used by this run for later reproduction.
    snapshot = run / "split_snapshot"
    snapshot.mkdir()
    for filename in ("train.csv", "val.csv", "test.csv", "labels.json", "split.json"):
        shutil.copyfile(args.split / filename, snapshot / filename)
    return run


def finish_run(run):
    path = run / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["status"] = "complete"
    save_json(path, meta)


def add_shared_arguments(parser):
    parser.add_argument("--split", type=Path, default=ROOT / "data/split/scaffold_42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test after configuration selection")
