#!/usr/bin/env python3
"""
download_resources.py
=====================
Downloads the complete resources/ folder (videos, model weights, recordings)
from a shared Google Drive zip archive.

Usage (after cloning)
---------------------
    python download_resources.py

Requirements
------------
    pip install gdown tqdm
"""
import sys
import zipfile
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
# Share the zip on Google Drive with "Anyone with the link can view",
# then paste the file ID here (the long string in the share URL).
#
# Share URL looks like:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#                                    ^^^^^^^^
GDRIVE_FILE_ID = "PASTE_YOUR_GDRIVE_FILE_ID_HERE"

ROOT     = Path(__file__).resolve().parent
ZIP_DEST = ROOT / "resources.zip"
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("  gdown not found – installing …")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "gdown"], check=True)
        import gdown
        return gdown


def main():
    if GDRIVE_FILE_ID == "PASTE_YOUR_GDRIVE_FILE_ID_HERE":
        print("ERROR: set GDRIVE_FILE_ID in download_resources.py before running.")
        print("  1. Upload resources.zip to Google Drive")
        print("  2. Right-click → Share → Anyone with the link")
        print("  3. Copy the file ID from the share URL")
        print("  4. Paste it into GDRIVE_FILE_ID at the top of this file")
        sys.exit(1)

    print("=" * 56)
    print("  Downloading resources.zip from Google Drive …")
    print(f"  File ID : {GDRIVE_FILE_ID}")
    print(f"  Dest    : {ZIP_DEST}")
    print("=" * 56)

    gdown = _ensure_gdown()

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    try:
        gdown.download(url, str(ZIP_DEST), quiet=False, fuzzy=True)
    except Exception as e:
        print(f"\nERROR: download failed: {e}")
        print("  Make sure the file is shared as 'Anyone with the link can view'.")
        sys.exit(1)

    if not ZIP_DEST.exists():
        print("ERROR: zip not downloaded (gdown returned no error but file is missing).")
        sys.exit(1)

    print(f"\n  Extracting to {ROOT} …")
    with zipfile.ZipFile(ZIP_DEST, "r") as zf:
        members = zf.namelist()
        print(f"  {len(members)} entries …", end=" ", flush=True)
        zf.extractall(ROOT)
    print("done")

    ZIP_DEST.unlink()
    print("  Cleaned up resources.zip")

    print("\n" + "=" * 56)
    print("  resources/ is ready.")
    print("=" * 56)


if __name__ == "__main__":
    main()
