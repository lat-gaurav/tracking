#!/usr/bin/env python3
# https://drive.google.com/file/d/1ujck7cSZuS5uPURZP7jhI0mxXCyEvjkt/view?usp=sharing
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
import os
import sys
import zipfile
from pathlib import Path

# ── Ensure the repo venv site-packages is on sys.path ─────────────────────────
_ROOT_EARLY = Path(__file__).resolve().parent
_VENV_SITE  = _ROOT_EARLY / "tracking" / "lib"
if _VENV_SITE.exists():
    for _sp in sorted(_VENV_SITE.rglob("site-packages"), key=lambda p: len(p.parts)):
        if str(_sp) not in sys.path:
            sys.path.insert(0, str(_sp))
        break  # only the first (shallowest) match
# ─────────────────────────────────────────────────────────────────────────────

# ── Config ────────────────────────────────────────────────────────────────────
# Share the zip on Google Drive with "Anyone with the link can view",
# then paste the file ID here (the long string in the share URL).
#
# Share URL looks like:
#   https://drive.google.com/file/d/<FILE_ID>/view?usp=sharing
#                                    ^^^^^^^^
GDRIVE_FILE_ID = "1ujck7cSZuS5uPURZP7jhI0mxXCyEvjkt"

ROOT     = Path(__file__).resolve().parent
ZIP_DEST = ROOT / "resources.zip"
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        print("  gdown not found – installing into venv …")
        import subprocess, importlib, site
        venv_pip = ROOT / "tracking" / "bin" / "pip"
        pip_exe  = str(venv_pip) if venv_pip.exists() else sys.executable
        cmd = [pip_exe, "install", "--quiet", "gdown"] if venv_pip.exists() \
              else [sys.executable, "-m", "pip", "install", "--quiet", "gdown"]
        subprocess.run(cmd, check=True)
        # make the freshly installed package visible in this process
        importlib.invalidate_caches()
        for sp in site.getsitepackages():
            if sp not in sys.path:
                sys.path.insert(0, sp)
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

    url = f"https://drive.google.com/file/d/{GDRIVE_FILE_ID}/view?usp=sharing"
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
