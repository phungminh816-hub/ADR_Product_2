"""Download, prepare and split molecular datasets."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Raw tables and model runs use the existing on-disk spelling "onside".
DATASET_FOLDERS = {"sider": "sider", "onsides": "onside"}


def source_path(path):
    resolved = Path(path).resolve()
    return resolved.relative_to(ROOT).as_posix() if resolved.is_relative_to(ROOT) else str(resolved)
