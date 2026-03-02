#!/usr/bin/env python3
"""
download_resources.py
=====================
Downloads all resources needed for this project:

  1. resources.zip  – model weights + small test videos (Google Drive)
  2. VisDrone-MOT   – Task 4 multi-object tracking dataset  (~11 GB, Google Drive)
  3. DroneCrowd     – crowd counting + tracking dataset     (~10 GB, Google Drive)

Dataset references
------------------
  VisDrone  : https://github.com/VisDrone/VisDrone-Dataset
  DroneCrowd: https://github.com/VisDrone/DroneCrowd
  M3OT      : must be requested from authors (not auto-downloaded)

Usage
-----
    python download_resources.py                 # everything
    python download_resources.py --skip-datasets # weights/videos only
    python download_resources.py --only visdrone-mot
    python download_resources.py --only dronecrowd

Requirements
------------
    pip install gdown
"""
import argparse
import subprocess
import sys
import zipfile
import importlib
import site
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ── Google Drive file IDs ─────────────────────────────────────────────────────
RESOURCES_ZIP_ID = "1ujck7cSZuS5uPURZP7jhI0mxXCyEvjkt"   # weights + test videos

VISDRONE_MOT = [
    ("VisDrone-MOT train (7.5 GB)",    "1-qX2d-P1Xr64ke6nTdlm33om1VxCUTSh", "resources/visdrone_mot/VisDrone2019-MOT-train.zip"),
    ("VisDrone-MOT val (1.5 GB)",      "1rqnKe9IgU_crMaxRoel9_nuUsMEBBVQu",  "resources/visdrone_mot/VisDrone2019-MOT-val.zip"),
    ("VisDrone-MOT test-dev (2.1 GB)", "14z8Acxopj1d86-qhsF1NwS4Bv3KYa4Wu", "resources/visdrone_mot/VisDrone2019-MOT-test-dev.zip"),
]

DRONECROWD = [
    ("DroneCrowd train (7.7 GB)",  "1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd",  "resources/dronecrowd/train_data-001.zip"),
    ("DroneCrowd test (2.6 GB)",   "1HY3V4QObrVjzXUxL_J86oxn2bi7FMUgd",  "resources/dronecrowd/test_data-002.zip"),
    # Note: DroneCrowd is officially distributed via Google Drive share link
    # from https://github.com/VisDrone/DroneCrowd — update IDs if stale.
]
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        pass

    print("  gdown not found – attempting install …")
    strategies = [
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "gdown"],
        [sys.executable, "-m", "pip", "install", "--quiet", "--user",
         "--break-system-packages", "gdown"],
        [sys.executable, "-m", "pip", "install", "--quiet", "gdown"],
    ]
    for cmd in strategies:
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            break
    else:
        print("\nERROR: could not auto-install gdown.")
        print("  pip install gdown")
        sys.exit(1)

    importlib.invalidate_caches()
    for sp in [site.getusersitepackages()] + site.getsitepackages():
        if sp not in sys.path:
            sys.path.insert(0, sp)
    import gdown
    return gdown


def _gdrive_download(gdown, file_id: str, dest: Path, label: str):
    """Download a single Google Drive file if not already present."""
    if dest.exists():
        print(f"  [skip] {label} – already downloaded ({dest.name})")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n  Downloading {label} …")
    url = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
    try:
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"  ERROR: {e}")
        return
    if not dest.exists():
        print(f"  ERROR: {dest.name} missing after download.")


def _extract_zip(zip_path: Path, extract_to: Path, remove_after=False):
    """Extract a zip if the expected directory doesn't exist yet."""
    if not zip_path.exists():
        print(f"  [skip extract] {zip_path.name} not found")
        return
    print(f"  Extracting {zip_path.name} → {extract_to} …", end=" ", flush=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("done")
    if remove_after:
        zip_path.unlink()
        print(f"  Removed {zip_path.name}")


def download_resources_zip(gdown):
    zip_dest = ROOT / "resources.zip"
    _gdrive_download(gdown, RESOURCES_ZIP_ID, zip_dest, "resources.zip (weights + test videos)")
    if zip_dest.exists():
        _extract_zip(zip_dest, ROOT, remove_after=True)


def download_visdrone_mot(gdown):
    print("\n── VisDrone-MOT ──────────────────────────────────────")
    for label, fid, rel_dest in VISDRONE_MOT:
        dest = ROOT / rel_dest
        _gdrive_download(gdown, fid, dest, label)
        # Extract alongside the zip
        split_name = dest.stem.replace("VisDrone2019-MOT-", "")
        extracted  = dest.parent / f"VisDrone2019-MOT-{split_name}"
        if dest.exists() and not extracted.exists():
            _extract_zip(dest, dest.parent)


def download_dronecrowd(gdown):
    print("\n── DroneCrowd ────────────────────────────────────────")
    for label, fid, rel_dest in DRONECROWD:
        dest = ROOT / rel_dest
        _gdrive_download(gdown, fid, dest, label)
        split_dir = dest.parent / dest.stem.split("-")[0]  # train_data / test_data
        if dest.exists() and not split_dir.exists():
            _extract_zip(dest, dest.parent)


def main():
    parser = argparse.ArgumentParser(description="Download project resources")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--skip-datasets", action="store_true",
                       help="Download weights/videos only, skip large datasets")
    group.add_argument("--only", choices=["visdrone-mot", "dronecrowd"],
                       help="Download only the specified dataset")
    args = parser.parse_args()

    gdown = _ensure_gdown()

    print("=" * 56)
    print("  Downloading project resources")
    print("=" * 56)

    if args.only == "visdrone-mot":
        download_visdrone_mot(gdown)
    elif args.only == "dronecrowd":
        download_dronecrowd(gdown)
    else:
        download_resources_zip(gdown)
        if not args.skip_datasets:
            download_visdrone_mot(gdown)
            download_dronecrowd(gdown)
            print("\n  Note: M3OT dataset must be requested from authors.")
            print("  See https://github.com/VisDrone/M3OT for access instructions.")

    print("\n" + "=" * 56)
    print("  Done.")
    print("=" * 56)


if __name__ == "__main__":
    main()
