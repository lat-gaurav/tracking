#!/usr/bin/env python3
"""
DroneCrowd → annotated MP4 visualizer
Reads per-sequence CVAT XML annotations and renders each video clip with
bounding boxes labeled #<track_id>.  All targets are "human".

Dataset layout:
  resources/dronecrowd/
    DroneCrowd/
      annotations/          ← 00001.xml … 00112.xml  (one XML per sequence)
      trainlist.txt
      testlist.txt
    train_data/images/      ← img001001.jpg … img082300.jpg
    test_data/images/       ← img011001.jpg … img040300.jpg

Image naming: img{seqID_3d}{frID_3d}.jpg   (seqID_5d "00001" → 3d "001")
XML  frame attr is 0-indexed; image frID is 1-indexed  (frame 0 → img…001.jpg)

Usage:
  python3 scratch/dronecrowd_to_video.py                 # all splits
  python3 scratch/dronecrowd_to_video.py --split train
  python3 scratch/dronecrowd_to_video.py --split test
  python3 scratch/dronecrowd_to_video.py --fps 30 --workers 6
  python3 scratch/dronecrowd_to_video.py --seq 00001 00005
"""

import argparse
import colorsys
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from xml.etree import ElementTree as ET

import cv2

# ---------------------------------------------------------------------------
ROOT     = Path(__file__).resolve().parent.parent
DC_ROOT  = ROOT / "resources" / "dronecrowd"
ANN_DIR  = DC_ROOT / "DroneCrowd" / "annotations"
OUT_ROOT = ROOT / "resources" / "video_test" / "dronecrowd_annotated"

SPLITS = {
    "train": (DC_ROOT / "DroneCrowd" / "trainlist.txt", DC_ROOT / "train_data" / "images"),
    "test":  (DC_ROOT / "DroneCrowd" / "testlist.txt",  DC_ROOT / "test_data"  / "images"),
}

# ---------------------------------------------------------------------------

def _track_color(track_id: int) -> tuple:
    """Deterministic distinct BGR color per track_id."""
    golden_ratio = 0.618033988749895
    h = (track_id * golden_ratio) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _load_xml(xml_path: Path) -> dict[int, list]:
    """Return frame_idx (0-based) → list of (track_id, x1, y1, x2, y2)."""
    frames: dict[int, list] = defaultdict(list)
    if not xml_path.exists():
        return frames
    tree = ET.parse(xml_path)
    for track in tree.getroot().findall("track"):
        tid = int(track.get("id", 0))
        for box in track.findall("box"):
            if box.get("outside", "0") == "1":
                continue
            fr  = int(box.get("frame"))
            x1  = int(float(box.get("xtl")))
            y1  = int(float(box.get("ytl")))
            x2  = int(float(box.get("xbr")))
            y2  = int(float(box.get("ybr")))
            frames[fr].append((tid, x1, y1, x2, y2))
    return frames


def _draw_label(img, text: str, x: int, y: int, color: tuple):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    lx = max(x, 0)
    ly = max(y - th - baseline - 1, 0)
    # tiny filled background
    cv2.rectangle(img, (lx, ly), (lx + tw + 2, ly + th + baseline + 1), color, -1)
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(img, text, (lx + 1, ly + th), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def _hud(img, split: str, seq: str, frame_no: int, total: int, n_people: int):
    h, w = img.shape[:2]
    bar_h = 26
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    text = (f"[DroneCrowd/{split}]  seq {seq}"
            f"   frame {frame_no:03d}/{total:03d}   people: {n_people}")
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.44, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------

def render_sequence(task):
    split, seq5d, img_dir_str, fps, out_dir_str = task
    img_dir = Path(img_dir_str)
    out_dir = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq3d    = f"{int(seq5d):03d}"
    xml_path = ANN_DIR / f"{seq5d}.xml"
    ann      = _load_xml(xml_path)

    # collect image paths sorted by frame number
    frame_paths = sorted(img_dir.glob(f"img{seq3d}*.jpg"),
                         key=lambda p: int(p.stem[6:]))  # digits after img###
    if not frame_paths:
        return (seq5d, False, "no images found")

    total   = len(frame_paths)
    out_path = out_dir / f"{split}_{seq5d}.mp4"

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        return (seq5d, False, "cannot read first frame")
    fh, fw = sample.shape[:2]

    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))

    for img_idx, fp in enumerate(frame_paths):
        # XML frame attr is 0-indexed; img filename 1-indexed
        xml_frame = img_idx          # frame 0 in XML = first image
        img = cv2.imread(str(fp))
        if img is None:
            continue

        objects = ann.get(xml_frame, [])
        for (tid, x1, y1, x2, y2) in objects:
            color = _track_color(tid)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            _draw_label(img, f"#{tid}", x1, y1, color)

        _hud(img, split, seq5d, img_idx + 1, total, len(objects))
        writer.write(img)

    writer.release()
    return (seq5d, True, f"{total} frames → {out_path.name}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="DroneCrowd → annotated MP4")
    parser.add_argument("--split",   default="all",
                        choices=["all", "train", "test"])
    parser.add_argument("--fps",     type=int, default=30)
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seq",     nargs="+", default=None,
                        help="Only render specific 5-digit sequence IDs e.g. --seq 00001 00005")
    args = parser.parse_args()

    splits_to_run = (list(SPLITS.items()) if args.split == "all"
                     else [(args.split, SPLITS[args.split])])

    tasks = []
    for split_name, (list_file, img_dir) in splits_to_run:
        if not list_file.exists():
            print(f"  [SKIP] list file not found: {list_file}")
            continue
        if not img_dir.exists():
            print(f"  [SKIP] image dir not found: {img_dir}")
            continue
        seq_ids = [l.strip() for l in list_file.read_text().splitlines() if l.strip()]
        if args.seq:
            seq_ids = [s for s in seq_ids if s in args.seq]
        out_dir = OUT_ROOT / split_name
        for seq5d in seq_ids:
            tasks.append((split_name, seq5d, str(img_dir), args.fps, str(out_dir)))

    if not tasks:
        print("No tasks — check that train_data/ and test_data/ are extracted.")
        return

    print(f"Rendering {len(tasks)} sequences  fps={args.fps}  workers={args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_sequence, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            seq5d, success, msg = fut.result()
            status = "OK " if success else "ERR"
            print(f"  [{i:>3}/{len(tasks)}] {status}  {seq5d}  {msg}")
            ok += success
            err += not success

    print(f"\nDone: {ok} OK  {err} errors")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
