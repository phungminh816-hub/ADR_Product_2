"""Locate completed training runs and read their metadata."""

import json
from pathlib import Path

from data import DATASET_FOLDERS


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_folder(metric_dir, dataset, model):
    folder = DATASET_FOLDERS[dataset]
    return metric_dir / folder / f"{model}_res_{folder}"


def find_latest_run(folder):
    for metadata in sorted(folder.glob("*/metadata.json"), reverse=True):
        if load_json(metadata)["status"] == "complete":
            return metadata.parent
    return None
