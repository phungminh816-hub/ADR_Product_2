"""Download SIDER or the pinned OnSIDES CSV release with source checksums."""

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from data import DATASET_FOLDERS, ROOT

SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASET_FOLDERS, default="sider")
    parser.add_argument("--url", help="URL of a SIDER-format .csv.gz (SIDER only)")
    parser.add_argument("--output", type=Path, help="Directory to save the downloaded data")
    parser.add_argument("--force", action="store_true", help="Replace an existing SIDER download")
    args = parser.parse_args()
    folder = args.output or ROOT / "data/raw" / DATASET_FOLDERS[args.dataset]
    if args.dataset == "onsides":
        if args.url or args.force:
            parser.error("--url and --force apply only to SIDER.")
        from data.raw.onsides import download
        download(folder)
        return
    folder.mkdir(parents=True, exist_ok=True)
    archive, csv = folder / "sider.csv.gz", folder / "sider.csv"
    meta_path = folder / "source.json"
    if (archive.exists() or csv.exists()) and not args.force:
        raise FileExistsError("Raw data already exists. Use --force to replace it.")
    saved = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() and not args.force else None
    url = args.url or (saved["url"] if saved is not None else SIDER_URL)
    with TemporaryDirectory(dir=folder, prefix=".sider-") as stage:
        archive_part, csv_part = Path(stage) / archive.name, Path(stage) / csv.name
        request = urllib.request.Request(url, headers={"User-Agent": "ADR_Product2/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_part.open("wb") as out:
            shutil.copyfileobj(response, out)
        with gzip.open(archive_part, "rb") as source, csv_part.open("wb") as out:
            shutil.copyfileobj(source, out)
        metadata = {"url": url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": hashlib.sha256(archive_part.read_bytes()).hexdigest(),
                    "csv_sha256": hashlib.sha256(csv_part.read_bytes()).hexdigest()}
        if saved is not None and any(saved[key] != metadata[key] for key in ("archive_sha256", "csv_sha256")):
            raise ValueError("Downloaded SIDER data does not match source.json.")
        staged_meta = Path(stage) / meta_path.name
        if saved is None:
            staged_meta.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        csv_part.replace(csv)
        if saved is None:
            staged_meta.replace(meta_path)
        if args.force:
            archive.unlink(missing_ok=True)
    print(csv)


if __name__ == "__main__":
    main()
