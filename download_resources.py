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
GDRIVE_FILE_ID = "1ujck7cSZuS5uPURZP7jhI0mxXCyEvjkt"

ROOT     = Path(__file__).resolve().parent
ZIP_DEST = ROOT / "resources.zip"
# ─────────────────────────────────────────────────────────────────────────────


def _ensure_gdown():
    try:
        import gdown
        return gdown
    except ImportError:
        pass

    print("  gdown not found – attempting install …")
    import subprocess, importlib, site

    # Try install strategies in order until one works
    strategies = [
        [sys.executable, "-m", "pip", "install", "--quiet", "--user", "gdown"],
        [sys.executable, "-m", "pip", "install", "--quiet", "--user",
         "--break-system-packages", "gdown"],   # Homebrew / externally-managed Python
        [sys.executable, "-m", "pip", "install", "--quiet", "gdown"],
    ]
    installed = False
    for cmd in strategies:
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            installed = True
            break

    if not installed:
        print("\nERROR: could not auto-install gdown.")
        print("  Please install it manually, then re-run:")
        print("    pip install gdown          # inside your active venv/conda")
        print("    pipx install gdown         # if you use pipx")
        sys.exit(1)

    # make the freshly-installed package visible without restarting
    importlib.invalidate_caches()
    for sp in [site.getusersitepackages()] + site.getsitepackages():
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
