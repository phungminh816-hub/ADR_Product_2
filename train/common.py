"""Shared feature generation, split validation, metrics and run metadata."""

import hashlib
import importlib.metadata
import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score

from data import ROOT
from train.runs import load_json, run_folder


def save_json(path, value):
    path = Path(path)
    content = json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)
    tmp = NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False)
    staged = Path(tmp.name)
    try:
        with tmp:
            tmp.write(content)
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def checked_bytes(folder, filename, meta):
    content = (folder / filename).read_bytes()
    if hashlib.sha256(content).hexdigest() != meta["files"][filename]:
        raise ValueError(f"Split checksum mismatch: {filename}")
    return content


def dataset_from_labels(labels):
    # OnSIDES preprocessing names each target "MedDRA:<PT code> <term>".
    return "onsides" if labels and all(label.startswith("MedDRA:") for label in labels) else "sider"


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


def load_splits(folder, radius=2, bits=2048, parts=("train", "val", "test")):
    folder = folder.resolve()
    meta = load_json(folder / "split.json")
    labels = json.loads(checked_bytes(folder, "labels.json", meta))
    if not isinstance(labels, list) or not labels or not all(isinstance(label, str) and label for label in labels):
        raise ValueError("Expected a nonempty list of label names.")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate label names.")
    splits, seen = {}, set()
    for name in ("train", "val", "test"):
        selected = name in parts
        content = checked_bytes(folder, f"{name}.csv", meta)
        frame = pd.read_csv(BytesIO(content), usecols=None if selected else ["drug_id"], dtype={"drug_id": str})
        if frame.empty or frame.drug_id.isna().any():
            raise ValueError(f"Empty split or missing drug IDs: {name}")
        if frame.drug_id.duplicated().any() or seen & set(frame.drug_id):
            raise ValueError("Duplicate or overlapping drug IDs.")
        seen.update(frame.drug_id)
        if not selected:
            continue
        y = frame[labels].to_numpy(dtype=np.float64)
        mask = np.isfinite(y)
        if not np.isin(y[mask], [0, 1]).all() or np.isinf(y).any():
            raise ValueError("Observed labels must be binary.")
        splits[name] = (fingerprints(frame.smiles, radius, bits), np.nan_to_num(y).astype(np.float32), mask)
    if "train" in splits and np.any(splits["train"][2].sum(axis=0) == 0):
        raise ValueError("At least one label has no observed training samples.")
    return splits, labels, meta


def require_scaffold(meta):
    expected = dict.fromkeys(("train_val", "train_test", "val_test"), 0)
    if meta["strategy"] != "scaffold" or meta.get("scaffold_overlap") != expected:
        raise ValueError("A Bemis–Murcko split with zero scaffold overlap is required.")


def evaluate(y, probability, mask, labels, threshold=.5):
    if y.ndim != 2 or mask.shape != y.shape or len(labels) != y.shape[1]:
        raise ValueError("Target, mask and label dimensions must match.")
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
    predicted = probability[mask] >= threshold
    truth = y[mask]
    return {"samples": len(y), "macro_ap": float(np.mean(aps)) if aps else None,
            "macro_roc_auc": float(np.mean(rocs)) if rocs else None,
            "valid_ap_labels": len(aps), "valid_roc_labels": len(rocs),
            "micro_precision": float(precision_score(truth, predicted, zero_division=0)) if truth.size else None,
            "micro_recall": float(recall_score(truth, predicted, zero_division=0)) if truth.size else None,
            "micro_f1": float(f1_score(truth, predicted, zero_division=0)) if truth.size else None,
            "threshold": threshold, "per_label": per_label}


def create_run(model_type, args, labels, split_meta):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dataset_name = dataset_from_labels(labels)
    metric_dir = args.metric_dir or ROOT / "metric"
    run = run_folder(metric_dir, dataset_name, model_type) / f"{model_type}_{stamp}_seed{args.seed}"
    run.mkdir(parents=True, exist_ok=False)
    config = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    save_json(run / "config.json", config)
    save_json(run / "labels.json", labels)
    save_json(run / "split.json", split_meta)
    libraries = ("numpy", "pandas", "scikit-learn", "rdkit", "joblib")
    if model_type == "xgboost":
        libraries += ("xgboost", "scipy")
    versions = {name: importlib.metadata.version(name) for name in libraries}
    save_json(run / "metadata.json", {"model": model_type, "dataset": dataset_name, "created_at_utc": stamp,
                                      "versions": versions, "status": "running"})
    # Preserve the exact data used by this run for later reproduction.
    snapshot = run / "split_snapshot"
    snapshot.mkdir()
    for filename in ("train.csv", "val.csv", "test.csv", "labels.json"):
        (snapshot / filename).write_bytes(checked_bytes(args.split, filename, split_meta))
    save_json(snapshot / "split.json", split_meta)
    return run


def finish_run(run):
    path = run / "metadata.json"
    meta = load_json(path)
    meta["status"] = "complete"
    save_json(path, meta)


def add_shared_arguments(parser):
    parser.add_argument("--split", type=Path, default=ROOT / "data/split/scaffold_42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--bits", type=int, default=2048)
    parser.add_argument("--metric-dir", type=Path, help="Directory for checkpoints and metrics (default: metric/)")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test after configuration selection")
