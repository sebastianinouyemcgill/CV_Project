"""
Helpers for Colab: stage NYU data from Google Drive onto fast local disk (/content).

Avoid unzipping 200k+ files on Drive — extract or rsync once per session to /content instead.
"""

from __future__ import annotations

import shutil
import subprocess
import tarfile
from pathlib import Path

from config import resolve_data_root


def verify_dataset(data_root: Path) -> Path:
    data_root = resolve_data_root(data_root)
    required = ["nyu2_train.csv", "nyu2_test.csv", "nyu2_train", "nyu2_test"]
    missing = [name for name in required if not (data_root / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Dataset incomplete under {data_root}. Missing: {missing}"
        )
    return data_root


def stage_from_tar(tar_path: Path, dest: Path) -> Path:
    """Extract a single .tar.gz archive to Colab local disk (fast)."""
    tar_path = Path(tar_path)
    dest = Path(dest)
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)

    dest.mkdir(parents=True, exist_ok=True)
    print(f"Extracting {tar_path} -> {dest} (local disk; may take several minutes)...")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=dest)
    return verify_dataset(dest)


def stage_from_rsync(src: Path, dest: Path) -> Path:
    """Copy an already-extracted Drive folder to Colab local disk (once per session)."""
    src = resolve_data_root(Path(src))
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)

    if shutil.which("rsync") is None:
        print("rsync not found, falling back to cp -a...")
        subprocess.run(["cp", "-a", f"{src}/.", str(dest)], check=True)
    else:
        print(f"rsync {src} -> {dest} (once per Colab session)...")
        subprocess.run(
            ["rsync", "-a", "--info=progress2", f"{src}/", f"{dest}/"],
            check=True,
        )
    return verify_dataset(dest)
