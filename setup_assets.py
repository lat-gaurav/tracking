#!/usr/bin/env python3
"""
setup_assets.py
===============
Run once after cloning to download all large assets (videos, model weights)
that cannot be stored in the git repository.

Usage
-----
    python setup_assets.py               # download everything
    python setup_assets.py --videos      # only test videos
    python setup_assets.py --weights     # only SiamRPN + ultralytics models
    python setup_assets.py --dry-run     # print what would be downloaded
    python setup_assets.py --skip-existing   # skip files already on disk (default: on)
    python setup_assets.py --force       # re-download even if file exists

Requirements (all installed by the repo's own venv / pip)
-----------------------------------------------------------
    pip install yt-dlp gdown pyyaml requests tqdm ultralytics
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

# ── locate repo root (same folder as this script) ─────────────────────────────
ROOT     = Path(__file__).resolve().parent
MANIFEST = ROOT / "assets_manifest.yaml"
VENV_PY  = ROOT / "tracking" / "bin" / "python3"
PY       = str(VENV_PY) if VENV_PY.exists() else sys.executable


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run(*cmd, check=True):
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    return subprocess.run([str(c) for c in cmd], check=check)


def _ensure_deps():
    """Install any missing lightweight Python deps (yt-dlp, gdown, pyyaml, requests)."""
    need = []
    for pkg, imp in [("yt-dlp", "yt_dlp"), ("gdown", "gdown"),
                     ("pyyaml", "yaml"),   ("requests", "requests"),
                     ("tqdm",   "tqdm")]:
        try:
            __import__(imp)
        except ImportError:
            need.append(pkg)
    if need:
        print(f"\n  Installing missing deps: {' '.join(need)}")
        _run(PY, "-m", "pip", "install", "--quiet", *need)


def _load_manifest():
    import yaml
    if not MANIFEST.exists():
        print(f"ERROR: manifest not found at {MANIFEST}", file=sys.stderr)
        sys.exit(1)
    with open(MANIFEST) as f:
        return yaml.safe_load(f)


def _sizeof_mb(path):
    return Path(path).stat().st_size / 1e6


# ── Video download ─────────────────────────────────────────────────────────────

def download_video(entry, force=False):
    dest = ROOT / entry["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  [skip] {dest.name}  ({_sizeof_mb(dest):.1f} MB already on disk)")
        return True

    url = entry["url"]
    print(f"\n  → {dest.name}")
    print(f"    source: {url}")

    if entry.get("direct"):
        # Plain HTTP download via requests + tqdm
        import requests
        from tqdm import tqdm
        try:
            r = requests.get(url, stream=True, timeout=30,
                             headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            total = int(r.headers.get("content-length", 0))
            with open(dest, "wb") as f, tqdm(total=total, unit="B", unit_scale=True,
                                             desc=dest.name, ncols=80) as bar:
                for chunk in r.iter_content(chunk_size=1 << 16):
                    f.write(chunk)
                    bar.update(len(chunk))
            print(f"    ✓  saved to {dest}  ({_sizeof_mb(dest):.1f} MB)")
            return True
        except Exception as e:
            print(f"    ✗  failed: {e}")
            return False
    else:
        # yt-dlp download
        opts = entry.get("yt_opts", [
            "--format", "bestvideo[height<=720][ext=mp4]+bestaudio/best[height<=720]",
            "--merge-output-format", "mp4",
        ])
        cmd = ["yt-dlp"] + opts + ["-o", str(dest), url]
        result = _run(*cmd, check=False)
        if result.returncode == 0 and dest.exists():
            print(f"    ✓  saved to {dest}  ({_sizeof_mb(dest):.1f} MB)")
            return True
        else:
            print(f"    ✗  yt-dlp failed (exit {result.returncode})")
            print(f"       Manual download: {url}")
            return False


# ── pysot weight download ──────────────────────────────────────────────────────

def download_pysot_weight(entry, force=False):
    import gdown
    dest = ROOT / entry["dest"]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        print(f"  [skip] {dest}  ({_sizeof_mb(dest):.1f} MB already on disk)")
        return True

    gdrive_id = entry["gdrive"]
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"\n  → {dest}  ({entry.get('note', '')})")
    try:
        gdown.download(url, str(dest), quiet=False, fuzzy=True)
        if dest.exists():
            print(f"    ✓  saved ({_sizeof_mb(dest):.1f} MB)")
            return True
        else:
            raise RuntimeError("file not created")
    except Exception as e:
        print(f"    ✗  gdown failed: {e}")
        print(f"    Manual fallback:")
        print(f"      1. Open in browser: https://drive.google.com/open?id={gdrive_id}")
        print(f"      2. Save file to:    {dest}")
        print(f"      Baidu Yun: https://pan.baidu.com/s/1GB9-aTtjG57SebraVoBfuQ  (code: j9yb)")
        return False


# ── Ultralytics warm-download ──────────────────────────────────────────────────

def warmdown_ultralytics(entries, force=False):
    """
    Importing YOLO("<name>") auto-downloads from ultralytics if not present.
    We just trigger that so it happens at setup time, not at first run.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  [skip] ultralytics not installed – models will download on first use")
        return

    for entry in entries:
        fpath = entry["filename"]          # e.g. resources/models/yolov8n.pt
        fname = Path(fpath).name           # yolov8n.pt  (what ultralytics knows)
        dest  = ROOT / fpath
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists() and not force:
            print(f"  [skip] {fpath}  ({_sizeof_mb(dest):.1f} MB already on disk)")
            continue
        print(f"\n  → {fpath}  ({entry.get('note', '')})")
        try:
            YOLO(fname)   # auto-downloads by short name to ultralytics cache / CWD
            import shutil
            # candidate locations ultralytics may have saved to
            candidates = [
                Path(fname),                                                      # CWD
                Path.home() / ".cache" / "ultralytics" / "weights" / fname,      # cache
                Path.home() / ".config" / "Ultralytics" / fname,
            ]
            if not dest.exists():
                for c in candidates:
                    if c.exists():
                        shutil.copy(c, dest)
                        break
            if dest.exists():
                print(f"    ✓  {fpath}  ({_sizeof_mb(dest):.1f} MB)")
            else:
                print(f"    ✓  {fname} downloaded to ultralytics cache (not copied to {dest})")
        except Exception as e:
            print(f"    ✗  {e}")


# ── Create .gitkeep placeholders ───────────────────────────────────────────────

def ensure_gitkeep_dirs(manifest):
    dirs = set()
    for section in ("videos", "pysot_weights", "custom_models"):
        for entry in manifest.get(section, []):
            dirs.add((ROOT / entry["dest"]).parent)
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        gk = d / ".gitkeep"
        if not gk.exists():
            gk.touch()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Download all large assets for the tracking repo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--videos",        action="store_true", help="Download test videos only")
    ap.add_argument("--weights",       action="store_true", help="Download model weights only")
    ap.add_argument("--force",         action="store_true", help="Re-download even if file exists")
    ap.add_argument("--dry-run",       action="store_true", help="Print what would be downloaded")
    ap.add_argument("--no-ultralytics",action="store_true", help="Skip ultralytics warm-download")
    args = ap.parse_args()

    do_videos  = args.videos  or not args.weights
    do_weights = args.weights or not args.videos

    print("=" * 62)
    print("  Tracking repo – asset setup")
    print(f"  Repo root : {ROOT}")
    print(f"  Manifest  : {MANIFEST.name}")
    print("=" * 62)

    if args.dry_run:
        print("\n[DRY RUN – nothing will be downloaded]\n")

    _ensure_deps()
    manifest = _load_manifest()
    ensure_gitkeep_dirs(manifest)

    failures = []
    t0 = time.time()

    # ── Videos ────────────────────────────────────────────────────────────────
    if do_videos:
        print(f"\n{'─'*40}")
        print(" TEST VIDEOS")
        print(f"{'─'*40}")
        for entry in manifest.get("videos", []):
            if args.dry_run:
                dest = ROOT / entry["dest"]
                print(f"  would download → {dest}")
                continue
            ok = download_video(entry, force=args.force)
            if not ok:
                failures.append(entry["dest"])

    # ── pysot weights ─────────────────────────────────────────────────────────
    if do_weights:
        print(f"\n{'─'*40}")
        print(" PYSOT SIAMRPN WEIGHTS")
        print(f"{'─'*40}")
        for entry in manifest.get("pysot_weights", []):
            if args.dry_run:
                print(f"  would download → {ROOT / entry['dest']}")
                continue
            ok = download_pysot_weight(entry, force=args.force)
            if not ok:
                failures.append(entry["dest"])

        # ── ultralytics ───────────────────────────────────────────────────────
        if not args.no_ultralytics:
            print(f"\n{'─'*40}")
            print(" ULTRALYTICS YOLO MODELS")
            print(f"{'─'*40}")
            if not args.dry_run:
                warmdown_ultralytics(manifest.get("ultralytics", []),
                                     force=args.force)
            else:
                for e in manifest.get("ultralytics", []):
                    print(f"  would download → {e['filename']}")

    # ── Custom / private models ────────────────────────────────────────────────
    custom = manifest.get("custom_models", [])
    missing_custom = [e for e in custom if not (ROOT / e["dest"]).exists()]
    if missing_custom:
        print(f"\n{'─'*40}")
        print(" CUSTOM MODELS  (manual download required)")
        print(f"{'─'*40}")
        for entry in missing_custom:
            dest = ROOT / entry["dest"]
            print(f"  ✗  {dest}")
            print(f"     {entry.get('note', '')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*62}")
    if args.dry_run:
        print("  Dry run complete.  Run without --dry-run to download.")
    elif failures:
        print(f"  Done in {elapsed:.0f}s.  {len(failures)} item(s) failed:")
        for f in failures:
            print(f"    - {f}")
        print("  Update their URLs in assets_manifest.yaml and re-run.")
    else:
        print(f"  All assets ready.  ({elapsed:.0f}s)")
    print(f"{'='*62}")


if __name__ == "__main__":
    main()
