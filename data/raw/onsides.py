"""Download OnSIDES and convert single-ingredient US labels to SMILES/PT targets."""

import hashlib
import json
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from rdkit import Chem

VERSION = "v3.1.1"
URL = f"https://github.com/tatonetti-lab/onsides/releases/download/{VERSION}/onsides-{VERSION}.zip"
SHA256 = "7739c73336786c92e73ca383af92d3f2814caf5e94206e08475b86d5a5ec22b9"
TABLES = {
    "product_label": {"label_id", "source"},
    "product_adverse_effect": {"product_label_id", "effect_meddra_id", "label_section"},
    "product_to_rxnorm": {"label_id", "rxnorm_product_id"},
    "vocab_rxnorm_ingredient_to_product": {"ingredient_id", "product_id"},
    "vocab_rxnorm_ingredient": {"rxnorm_id", "rxnorm_name"},
    "vocab_meddra_adverse_effect": {"meddra_id", "meddra_name", "meddra_term_type"},
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(output, archive=None):
    """Extract only required CSVs; publish the directory after complete validation."""
    output = Path(output)
    if output.exists():
        if all((output / f"{table}.csv").is_file() for table in TABLES):
            print(f"Already downloaded: {output}")
            return
        raise FileExistsError(f"Incomplete OnSIDES data directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onsides-download-", dir=output.parent) as scratch:
        scratch = Path(scratch)
        archive_path = Path(archive) if archive else scratch / "release.zip"
        if not archive:
            print(f"Downloading {URL}", flush=True)
            request = urllib.request.Request(URL, headers={"User-Agent": "ADR_Product2/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, archive_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        if sha256(archive_path) != SHA256:
            raise ValueError("OnSIDES archive SHA-256 mismatch; expected the official v3.1.1 data ZIP.")
        staged = scratch / "tables"
        staged.mkdir()
        with zipfile.ZipFile(archive_path) as bundle:
            for table, required in TABLES.items():
                matches = [item for item in bundle.infolist()
                           if not item.is_dir() and Path(item.filename).name == f"{table}.csv"]
                if len(matches) != 1:
                    raise ValueError(f"Expected exactly one {table}.csv in release.")
                target = staged / f"{table}.csv"
                # Copy to fixed basenames rather than trusting paths in the ZIP.
                with bundle.open(matches[0]) as source, target.open("wb") as out:
                    shutil.copyfileobj(source, out)
                columns = pd.read_csv(target, nrows=0).columns
                if not required.issubset(columns):
                    raise ValueError(f"Unsupported OnSIDES schema: {table}: {list(columns)}")
        metadata = {"dataset": "onsides", "version": VERSION, "url": URL,
                    "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": SHA256,
                    "files": {path.name: sha256(path) for path in sorted(staged.glob("*.csv"))}}
        (staged / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        staged.rename(output)
    print(output)


def map_smiles(raw):
    """Look up exact ChEMBL names, saving each result so interrupted runs can resume."""
    target = raw / "ingredient_smiles.csv"
    saved = pd.read_csv(target, dtype=str, keep_default_na=False) if target.exists() else pd.DataFrame()
    if not saved.empty and saved.rxnorm_id.duplicated().any():
        raise ValueError("Duplicate IDs in ingredient_smiles.csv")
    cache = {row["rxnorm_id"]: row for row in saved.to_dict("records")}
    ingredients = pd.read_csv(raw / "vocab_rxnorm_ingredient.csv", dtype=str)
    partial = target.with_suffix(".csv.part")
    try:
        for row in ingredients.itertuples(index=False):
            if row.rxnorm_id in cache:
                if cache[row.rxnorm_id].get("rxnorm_name") != row.rxnorm_name:
                    raise ValueError(f"Cached ingredient name changed: {row.rxnorm_id}")
                continue
            url = "https://www.ebi.ac.uk/chembl/api/data/molecule.json?" + urllib.parse.urlencode(
                {"pref_name__iexact": row.rxnorm_name, "limit": 1000})
            with urllib.request.urlopen(url, timeout=60) as response:
                result = json.load(response)
            matches = result["molecules"]
            status, smiles, provider_id = "not_found", "", ""
            if result["page_meta"]["total_count"] > 1:
                status = "ambiguous"
            elif len(matches) == 1:
                smiles = (matches[0].get("molecule_structures") or {}).get("canonical_smiles") or ""
                molecule = Chem.MolFromSmiles(smiles) if smiles else None
                status = "ok" if molecule is not None and molecule.GetNumAtoms() else "invalid_smiles"
                smiles = Chem.MolToSmiles(molecule, isomericSmiles=True) if status == "ok" else ""
                provider_id = matches[0]["molecule_chembl_id"]
            cache[row.rxnorm_id] = {"rxnorm_id": row.rxnorm_id, "rxnorm_name": row.rxnorm_name,
                                   "smiles": smiles, "provider": "chembl", "provider_id": provider_id,
                                   "status": status, "query_url": url,
                                   "retrieved_at_utc": datetime.now(timezone.utc).isoformat()}
            pd.DataFrame(cache.values()).to_csv(partial, index=False)
            partial.replace(target)
            print(f"{len(cache)}/{len(ingredients)} {row.rxnorm_name}: {status}", flush=True)
            time.sleep(.35)
    finally:
        partial.unlink(missing_ok=True)
    return target


def prepare_onsides(raw, output, mapping=None):
    """Build benchmark labels: extracted association=1, unlisted association=0."""
    from data.proceed.prepare import prepare

    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    mapping = mapping or map_smiles(raw)

    def read(table):
        return pd.read_csv(raw / f"{table}.csv", dtype=str, keep_default_na=False)

    structures = pd.read_csv(mapping, dtype=str, keep_default_na=False)
    if not {"rxnorm_id", "smiles"}.issubset(structures) or structures.rxnorm_id.duplicated().any():
        raise ValueError("Mapping needs unique rxnorm_id and smiles columns.")
    vocabulary = read("vocab_rxnorm_ingredient")
    if "rxnorm_name" in structures:
        names = structures.merge(vocabulary, on="rxnorm_id", suffixes=("_mapping", "_raw"))
        if (names.rxnorm_name_mapping != names.rxnorm_name_raw).any():
            raise ValueError("Mapping ingredient names do not match OnSIDES.")
    if "status" in structures:
        structures = structures[structures.status.isin(["ok", "manual"])]
    smiles = {}
    known_ids = set(vocabulary.rxnorm_id)
    for row in structures.itertuples(index=False):
        molecule = Chem.MolFromSmiles(row.smiles) if row.smiles else None
        if row.rxnorm_id in known_ids and molecule is not None and molecule.GetNumAtoms():
            smiles[row.rxnorm_id] = Chem.MolToSmiles(molecule, isomericSmiles=True)
    links = read("product_to_rxnorm").merge(read("vocab_rxnorm_ingredient_to_product"), how="left",
                                           left_on="rxnorm_product_id", right_on="product_id")
    counts = links.groupby("label_id").ingredient_id.nunique()
    unmapped = set(links.loc[links.ingredient_id.isna(), "label_id"])
    single = set(counts.index[counts == 1]) - unmapped
    labels = read("product_label")
    single &= set(labels.loc[labels.source == "US", "label_id"])
    label_smiles = {row.label_id: smiles[row.ingredient_id] for row in links.itertuples(index=False)
                    if row.label_id in single and row.ingredient_id in smiles}
    effects = read("vocab_meddra_adverse_effect")
    names = dict(effects.loc[effects.meddra_term_type == "PT", ["meddra_id", "meddra_name"]].values)
    positives = {}
    for chunk in pd.read_csv(raw / "product_adverse_effect.csv", dtype=str, chunksize=100000,
                             usecols=["product_label_id", "effect_meddra_id", "label_section"]):
        chunk = chunk[(chunk.label_section == "AR") & chunk.product_label_id.isin(label_smiles)
                      & chunk.effect_meddra_id.isin(names)]
        for row in chunk.itertuples(index=False):
            positives.setdefault(label_smiles[row.product_label_id], set()).add(row.effect_meddra_id)
    if not positives:
        raise ValueError("No single-ingredient US AR/PT associations with valid SMILES.")
    codes = sorted(set().union(*positives.values()), key=int)
    labels = [f"MedDRA:{code} {names[code]}" for code in codes]
    records = [{"smiles": structure, **{label: int(code in observed) for code, label in zip(codes, labels)}}
               for structure, observed in sorted(positives.items())]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".onsides-", dir=output.parent) as scratch:
        source = Path(scratch) / "molecules.csv"
        pd.DataFrame(records).to_csv(source, index=False)
        staged = Path(scratch) / "processed"
        prepare(source, staged, labels)
        report = json.loads((staged / "report.json").read_text(encoding="utf-8"))
        report.update(source=str(raw.resolve()), dataset="onsides", mapping_sha256=sha256(mapping),
                      sources=["US"], section="AR", term_type="PT",
                      zero_means="association not recorded; not confirmed clinical absence")
        report["normalized_csv_sha256"] = report.pop("source_sha256")
        (staged / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        staged.rename(output)
    print(output)
