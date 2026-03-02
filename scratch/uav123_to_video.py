"""
uav123_to_video.py
------------------
Convert UAV123_10fps image sequences → annotated MP4 videos.

Usage:
  python3 scratch/uav123_to_video.py [--seq bird1_1] [--all] [--workers 4]

Outputs go to:  resources/video_test/uav123_annotated/<seqname>.mp4
"""

import re, zipfile, sys, argparse
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
import cv2

# ─── paths ───────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).resolve().parent.parent
UAV_DIR  = ROOT / "resources" / "uav123"
ZIP_PATH = UAV_DIR / "Dataset_UAV123_10fps.zip"
EXTRACT  = UAV_DIR / "UAV123_10fps"
OUT_DIR  = ROOT / "resources" / "video_test" / "uav123_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── category colours (BGR) ───────────────────────────────────────────────────
PALETTE = {
    "bike":      (0,   215, 255),   # gold
    "bird":      (255, 200,  50),   # sky blue
    "boat":      (200, 130,   0),   # teal
    "building":  (100, 180, 255),   # orange
    "car":       (0,   255, 100),   # green
    "group":     (180,   0, 255),   # purple
    "person":    (255,  80,  80),   # blue
    "truck":     (0,   100, 255),   # deep orange
    "uav":       (50,  255, 255),   # yellow
    "wakeboard": (200, 255,   0),   # cyan-green
}
DEFAULT_COLOR = (200, 200, 200)

def category_color(seq_name: str):
    for cat, col in PALETTE.items():
        if seq_name.startswith(cat):
            return col
    return DEFAULT_COLOR


# ─── parse configSeqs.m ──────────────────────────────────────────────────────
def parse_config(config_text: str) -> list[dict]:
    """Return list of dicts: name, folder, startFrame, endFrame, nz"""
    pattern = re.compile(
        r"struct\('name','(?P<name>[^']+)',"
        r"'path','[^']*[/\\](?P<folder>[^/\\\\']+)[/\\]',"
        r"'startFrame',(?P<start>\d+),"
        r"'endFrame',(?P<end>\d+),"
        r"'nz',(?P<nz>\d+)",
        re.DOTALL,
    )
    seqs = []
    for m in pattern.finditer(config_text):
        seqs.append(dict(
            name       = m.group("name"),
            folder     = m.group("folder"),
            start      = int(m.group("start")),
            end        = int(m.group("end")),
            nz         = int(m.group("nz")),
        ))
    return seqs


def load_config() -> list[dict]:
    config_text = (EXTRACT / "configSeqs.m").read_text()
    return parse_config(config_text)


# ─── annotation loader ───────────────────────────────────────────────────────
def load_anno(seq_name: str) -> list:
    """Return list of (x,y,w,h) or None per frame — relative to seq startFrame."""
    anno_path = EXTRACT / "anno" / "UAV123_10fps" / f"{seq_name}.txt"
    boxes = []
    with open(anno_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4 or "NaN" in line:
                boxes.append(None)
            else:
                try:
                    boxes.append(tuple(float(v) for v in parts[:4]))
                except ValueError:
                    boxes.append(None)
    return boxes


# ─── single sequence → mp4 ───────────────────────────────────────────────────
def render_sequence(seq: dict) -> str:
    name   = seq["name"]
    folder = seq["folder"]
    start  = seq["start"]
    end    = seq["end"]
    nz     = seq["nz"]
    color  = category_color(name)

    out_path = OUT_DIR / f"{name}.mp4"
    if out_path.exists():
        return f"[skip]  {name}  (already exists)"

    img_dir = EXTRACT / "data_seq" / "UAV123_10fps" / folder
    if not img_dir.exists():
        return f"[ERROR] {name}: image dir missing ({img_dir})"

    try:
        boxes = load_anno(name)
    except FileNotFoundError:
        return f"[ERROR] {name}: annotation file missing"

    # collect frame paths in range [start, end] inclusive
    frames = []
    for fn in sorted(img_dir.glob("*.jpg")):
        idx = int(fn.stem)
        if start <= idx <= end:
            frames.append((idx, fn))
    frames.sort()

    if not frames:
        return f"[ERROR] {name}: no frames found in [{start},{end}]"

    # probe resolution from first frame
    sample = cv2.imread(str(frames[0][1]))
    if sample is None:
        return f"[ERROR] {name}: couldn't read first frame"
    h, w = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, 10.0, (w, h))

    for rel_i, (idx, fpath) in enumerate(frames):
        img = cv2.imread(str(fpath))
        if img is None:
            continue

        # bounding box
        box = boxes[rel_i] if rel_i < len(boxes) else None
        if box is not None:
            x, y, bw, bh = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 2)
            # target centre dot
            cx, cy = x + bw // 2, y + bh // 2
            cv2.circle(img, (cx, cy), 3, color, -1)
            # size label
            cv2.putText(img, f"{bw}x{bh}", (x, max(y - 4, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

        # sequence name + frame counter (top-left HUD)
        hud = f"{name}  fr:{idx:0{nz}d}  ({rel_i+1}/{len(frames)})"
        cv2.rectangle(img, (0, 0), (len(hud) * 8 + 8, 20), (0, 0, 0), -1)
        cv2.putText(img, hud, (4, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    return f"[done]  {name}  → {out_path.name}  ({len(frames)} frames)"


# ─── extraction helper ───────────────────────────────────────────────────────
def extract_if_needed():
    if EXTRACT.exists() and any(EXTRACT.iterdir()):
        print(f"[extract] Already extracted at {EXTRACT}")
        return
    if not ZIP_PATH.exists():
        sys.exit(f"[ERROR] Zip not found: {ZIP_PATH}")
    print(f"[extract] Extracting {ZIP_PATH.name}  (~4.4 GB) → {UAV_DIR} ...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        members = zf.namelist()
        total   = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, UAV_DIR)
            if i % 5000 == 0 or i == total:
                print(f"  {i}/{total} files extracted", end="\r")
    print("\n[extract] Done.")


# ─── main ────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="UAV123 → annotated MP4")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--all",   action="store_true", help="Convert all 124 sequences")
    grp.add_argument("--seq",   nargs="+",           help="Specific sequence name(s), e.g. person1 car6_1")
    grp.add_argument("--list",  action="store_true", help="List all available sequences and exit")
    parser.add_argument("--workers", type=int, default=min(4, cpu_count()),
                        help="Parallel workers (default: min(4, cpu_count))")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction check (assume already extracted)")
    args = parser.parse_args()

    if not args.no_extract:
        extract_if_needed()

    seqs = load_config()
    print(f"[config] Loaded {len(seqs)} sequences from configSeqs.m")

    if args.list:
        for s in seqs:
            print(f"  {s['name']:<20}  folder={s['folder']:<12}  "
                  f"frames {s['start']:>5}–{s['end']:>5}  "
                  f"({s['end']-s['start']+1} fr)")
        return

    if args.seq:
        names   = set(args.seq)
        to_proc = [s for s in seqs if s["name"] in names]
        missing = names - {s["name"] for s in to_proc}
        if missing:
            print(f"[warn] Unknown sequences: {missing}")
    else:
        to_proc = seqs

    print(f"[render] Processing {len(to_proc)} sequences with {args.workers} workers …")

    if args.workers > 1 and len(to_proc) > 1:
        with Pool(args.workers) as pool:
            for msg in pool.imap_unordered(render_sequence, to_proc, chunksize=1):
                print(msg)
    else:
        for seq in to_proc:
            print(render_sequence(seq))

    print(f"\n[done] Videos saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
