"""Download SIDER or the pinned OnSIDES CSV release with source checksums."""

import argparse
import gzip
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIDER_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/sider.csv.gz"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["sider", "onsides"], default="sider")
    parser.add_argument("--url", default=SIDER_URL, help="URL of a SIDER-format .csv.gz")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.dataset == "onsides":
        from data.raw.onsides import download
        download(ROOT / "data/raw/onsides")
        return
    folder = ROOT / "data/raw"
    archive, csv = folder / "sider.csv.gz", folder / "sider.csv"
    meta_path = folder / "source.json"
    if (archive.exists() or csv.exists() or meta_path.exists()) and not args.force:
        raise FileExistsError("Raw data already exists. Use --force to replace it.")
    archive_part, csv_part = folder / "sider.csv.gz.part", folder / "sider.csv.part"
    try:
        request = urllib.request.Request(args.url, headers={"User-Agent": "ADR_Product2/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response, archive_part.open("wb") as out:
            shutil.copyfileobj(response, out)
        with gzip.open(archive_part, "rb") as source, csv_part.open("wb") as out:
            shutil.copyfileobj(source, out)
        metadata = {"url": args.url, "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
                    "archive_sha256": hashlib.sha256(archive_part.read_bytes()).hexdigest(),
                    "csv_sha256": hashlib.sha256(csv_part.read_bytes()).hexdigest()}
        archive_part.replace(archive)
        csv_part.replace(csv)
        meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        archive_part.unlink(missing_ok=True)
        csv_part.unlink(missing_ok=True)
    print(csv)


if __name__ == "__main__":
    main()
