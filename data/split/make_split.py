"""Create random or Bemis-Murcko scaffold splits and immutable data snapshots."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

ROOT = Path(__file__).resolve().parents[2]


def make_split(processed, output, strategy="scaffold", seed=42, train_ratio=.8, val_ratio=.1,
               max_labels=None):
    if max_labels is not None and max_labels <= 0:
        raise ValueError("--max-labels must be positive.")
    if not 0 < train_ratio < 1 or not 0 < val_ratio < 1 or train_ratio + val_ratio >= 1:
        raise ValueError("Ratios must leave nonempty train, validation and test fractions.")
    frame = pd.read_csv(processed / "processed.csv")
    labels = json.loads((processed / "labels.json").read_text(encoding="utf-8"))
    if frame.drug_id.duplicated().any() or len(frame) < 3:
        raise ValueError("Need at least three unique molecules.")
    scaffolds = [MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s)) for s in frame.smiles]
    rng = np.random.default_rng(seed)
    if strategy == "random":
        order = rng.permutation(len(frame))
        a, b = int(len(frame) * train_ratio), int(len(frame) * val_ratio)
        parts = [order[:a], order[a:a+b], order[a+b:]]
    elif strategy == "scaffold":
        groups = {}
        for index, scaffold in enumerate(scaffolds):
            groups.setdefault(scaffold, []).append(index)
        if len(groups) < 3:
            raise ValueError("Scaffold split needs at least three distinct scaffold groups.")
        grouped = [groups[key] for key in sorted(groups)]
        rng.shuffle(grouped)
        grouped.sort(key=len, reverse=True)
        targets = len(frame) * np.array([train_ratio, val_ratio, 1-train_ratio-val_ratio])
        parts = [[], [], []]
        for group in grouped:
            destination = int(np.argmax(targets - np.array([len(part) for part in parts])))
            parts[destination].extend(group)
    else:
        raise ValueError("Unknown split strategy.")
    if any(len(part) == 0 for part in parts):
        raise ValueError("Empty split. Adjust ratios, seed or grouping.")
    if max_labels is not None:
        training = frame.iloc[parts[0]]
        positive = (training[labels] == 1).sum()
        negative = (training[labels] == 0).sum()
        candidates = [label for label in labels if positive[label] > 0 and negative[label] > 0]
        candidates.sort(key=lambda label: (-int(positive[label]), label))
        labels = candidates[:max_labels]
        if not labels:
            raise ValueError("No labels have both positive and negative training samples.")
    output.mkdir(parents=True, exist_ok=False)
    metadata = {"strategy": strategy, "seed": seed,
                "ratios": [train_ratio, val_ratio, 1-train_ratio-val_ratio],
                "processed_sha256": hashlib.sha256((processed / "processed.csv").read_bytes()).hexdigest(),
                "files": {}, "counts": {}, "label_counts": {}, "scaffold_overlap": {}}
    if max_labels is not None:
        metadata["label_selection"] = {"based_on": "train only", "max_labels": max_labels}
    scaffold_sets = {}
    for name, indices in zip(("train", "val", "test"), parts):
        subset = frame.iloc[indices][["drug_id", "smiles", *labels]]
        subset.to_csv(output / f"{name}.csv", index=False)
        metadata["files"][f"{name}.csv"] = hashlib.sha256((output / f"{name}.csv").read_bytes()).hexdigest()
        metadata["counts"][name] = len(subset)
        metadata["label_counts"][name] = {
            label: {"positive": int((subset[label] == 1).sum()),
                    "negative": int((subset[label] == 0).sum()),
                    "missing": int(subset[label].isna().sum())} for label in labels}
        scaffold_sets[name] = {scaffolds[i] for i in indices}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        metadata["scaffold_overlap"][f"{a}_{b}"] = len(scaffold_sets[a] & scaffold_sets[b])
    (output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    metadata["files"]["labels.json"] = hashlib.sha256((output / "labels.json").read_bytes()).hexdigest()
    (output / "split.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"strategy": strategy, "counts": metadata["counts"],
                      "scaffold_overlap": metadata["scaffold_overlap"]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/proceed/sider")
    parser.add_argument("--strategy", choices=["random", "scaffold"], default="scaffold")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=.8)
    parser.add_argument("--val-ratio", type=float, default=.1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-labels", type=int, help="Keep most frequent labels using TRAIN counts only")
    args = parser.parse_args()
    output = args.output or ROOT / f"data/split/{args.strategy}_{args.seed}"
    make_split(args.processed, output, args.strategy, args.seed, args.train_ratio, args.val_ratio,
               args.max_labels)


if __name__ == "__main__":
    main()
