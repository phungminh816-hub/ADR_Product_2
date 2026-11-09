"""Validate binary labels, canonicalize molecules and merge compatible duplicates."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

ROOT = Path(__file__).resolve().parents[2]


def prepare(source, output, label_columns=None):
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    frame = pd.read_csv(source)
    if "smiles" not in frame:
        raise ValueError("CSV must contain a smiles column.")
    labels = label_columns or [name for name in frame if name != "smiles"]
    if not labels or not set(labels).issubset(frame.columns):
        raise ValueError("No valid label columns.")
    for name in labels:
        observed = frame[name].dropna()
        if not observed.isin([0, 1]).all():
            raise ValueError(f"Label {name} must contain only 0, 1 or missing values.")
    records, rejected = [], []
    for index, row in frame.iterrows():
        smiles = row["smiles"]
        mol = Chem.MolFromSmiles(smiles.strip()) if isinstance(smiles, str) else None
        if mol is None or mol.GetNumAtoms() == 0:
            rejected.append({"row": int(index), "smiles": str(smiles)})
            continue
        canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
        records.append({"drug_id": hashlib.sha256(canonical.encode()).hexdigest(),
                        "smiles": canonical, **{name: row[name] for name in labels}})
    if not records:
        raise ValueError("No valid molecules.")
    valid = pd.DataFrame(records)
    for drug_id, group in valid.groupby("drug_id"):
        conflicting = [name for name in labels if group[name].nunique() > 1]
        if conflicting:
            raise ValueError(f"Conflicting duplicate labels for {drug_id}: {conflicting}")
    # first() takes the first observed value, preserving complementary missing labels.
    clean = valid.groupby("drug_id", sort=False, as_index=False).first()
    if clean[labels].notna().sum().sum() == 0:
        raise ValueError("Dataset has no observed labels.")
    output.mkdir(parents=True, exist_ok=False)
    clean.to_csv(output / "processed.csv", index=False)
    (output / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")
    report = {"source": str(source.resolve()),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "raw_rows": len(frame), "processed_rows": len(clean), "labels": labels,
              "invalid_smiles": rejected, "merged_rows": len(valid) - len(clean),
              "missing_labels": int(clean[labels].isna().sum().sum()),
              "standardization": "canonical isomeric SMILES; salts and tautomers preserved"}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("raw_rows", "processed_rows", "merged_rows", "missing_labels")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sider", "onsides"], default="sider")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--labels", nargs="+", help="Explicit label columns for a CSV with metadata")
    parser.add_argument("--smiles-map", type=Path, help="Use an existing OnSIDES rxnorm_id/smiles mapping")
    args = parser.parse_args()
    if args.dataset == "onsides":
        from data.raw.onsides import prepare_onsides
        raw = args.input or ROOT / "data/raw/onsides"
        prepare_onsides(raw, args.output or ROOT / "data/proceed/onsides", args.smiles_map)
    else:
        prepare(args.input or ROOT / "data/raw/sider.csv",
                args.output or ROOT / "data/proceed/sider", args.labels)


if __name__ == "__main__":
    main()
