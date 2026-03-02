#!/usr/bin/env python3
"""
VisDrone-MOT → annotated MP4 visualizer
Renders each sequence as an MP4 with bounding boxes labeled <class>#<track_id>.

Dataset layout (Task 4 – Multi-Object Tracking):
  resources/visdrone_mot/VisDrone2019-MOT-{train,val,test-dev}/
    sequences/<seqname>/   ← consecutive JPEGs (0000001.jpg …)
    annotations/<seqname>.txt  ← MOT-style CSV (see below)

Annotation columns (1-indexed):
  frame, track_id, x, y, w, h, score, category, truncation, occlusion
  category 0 = ignored region (skipped)

Usage:
  python3 scratch/visdrone_mot_to_video.py               # all splits
  python3 scratch/visdrone_mot_to_video.py --split train
  python3 scratch/visdrone_mot_to_video.py --split val
  python3 scratch/visdrone_mot_to_video.py --split test-dev
  python3 scratch/visdrone_mot_to_video.py --fps 20 --workers 4
"""

import argparse
import colorsys
import csv
import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
MOT_ROOT = ROOT / "resources" / "visdrone_mot"
OUT_ROOT = ROOT / "resources" / "video_test" / "visdrone_mot_annotated"

SPLITS = {
    "train":    "VisDrone2019-MOT-train",
    "val":      "VisDrone2019-MOT-val",
    "test-dev": "VisDrone2019-MOT-test-dev",
}

# ---------------------------------------------------------------------------
# Category info
# ---------------------------------------------------------------------------
CAT_NAMES = {
    0:  "ignored",
    1:  "pedestrian",
    2:  "people",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tri",
    9:  "bus",
    10: "motor",
}

# Base BGR colors per category
CAT_COLORS = {
    0:  (80,  80,  80),    # ignored  – dark grey (never drawn)
    1:  (255, 128,   0),   # pedestrian – blue-orange
    2:  (255,   0, 200),   # people – magenta
    3:  (0,   215, 255),   # bicycle – gold
    4:  (0,    60, 220),   # car – red
    5:  (0,   180,   0),   # van – green
    6:  (0,   200, 100),   # truck – lime
    7:  (220, 100,  50),   # tricycle – teal-ish
    8:  (180,  50, 220),   # awning-tri – purple
    9:  (0,   220, 220),   # bus – yellow
    10: (200,   0, 100),   # motor – indigo
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _track_color(base_bgr: tuple, track_id: int) -> tuple:
    """Shift hue slightly per track_id so same-class objects are distinguishable."""
    b, g, r = base_bgr
    h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    h = (h + (track_id * 0.07)) % 1.0
    nr, ng, nb = colorsys.hsv_to_rgb(h, max(0.5, s), max(0.6, v))
    return (int(nb * 255), int(ng * 255), int(nr * 255))


def _load_annotations(ann_path: Path) -> dict[int, list]:
    """Return dict: frame_idx → list of (track_id, x, y, w, h, category)."""
    frames: dict[int, list] = defaultdict(list)
    if not ann_path.exists():
        return frames
    with ann_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 8:
                continue
            frame_idx = int(row[0])
            track_id  = int(row[1])
            x, y, w, h = int(row[2]), int(row[3]), int(row[4]), int(row[5])
            category   = int(row[7])
            if category == 0:  # ignored region
                continue
            frames[frame_idx].append((track_id, x, y, w, h, category))
    return frames


def _draw_label(img, text: str, x: int, y: int, color: tuple):
    """Draw a filled-background label above the bbox."""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    lx = max(x, 0)
    ly = max(y - th - baseline - 2, 0)
    cv2.rectangle(img, (lx, ly), (lx + tw + 4, ly + th + baseline + 2), color, -1)
    # Choose white/black text for contrast
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(img, text, (lx + 2, ly + th + 1), font, font_scale, text_color, thickness, cv2.LINE_AA)


def _hud(img, split: str, seq: str, frame_idx: int, total: int, n_objects: int):
    """Draw semi-transparent HUD bar at top."""
    h, w = img.shape[:2]
    bar_h = 28
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"[VisDrone-MOT/{split}]  {seq}   frame {frame_idx:04d}/{total:04d}   objects: {n_objects}"
    cv2.putText(img, text, (6, 19), font, 0.46, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Per-sequence renderer
# ---------------------------------------------------------------------------

def render_sequence(args_tuple):
    split_name, split_dir_name, seq_name, fps, out_dir_str = args_tuple
    out_dir   = Path(out_dir_str)
    split_dir = MOT_ROOT / split_dir_name
    seq_dir   = split_dir / "sequences" / seq_name
    ann_path  = split_dir / "annotations" / f"{seq_name}.txt"
    out_path  = out_dir / f"{split_name}_{seq_name}.mp4"

    frames_ann  = _load_annotations(ann_path)
    frame_paths = sorted(seq_dir.glob("*.jpg"))

    if not frame_paths:
        return (seq_name, False, "no frames")

    out_dir.mkdir(parents=True, exist_ok=True)
    total = len(frame_paths)

    # Determine frame size from first image
    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        return (seq_name, False, "cannot read first frame")
    fh, fw = sample.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (fw, fh))

    for idx, fp in enumerate(frame_paths):
        frame_number = idx + 1          # VisDrone annotations are 1-indexed
        img = cv2.imread(str(fp))
        if img is None:
            continue

        objects = frames_ann.get(frame_number, [])

        for (track_id, x, y, w, h, category) in objects:
            base_color = CAT_COLORS.get(category, (200, 200, 200))
            color = _track_color(base_color, track_id)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            label = f"{CAT_NAMES.get(category, str(category))}#{track_id}"
            _draw_label(img, label, x, y, color)

        _hud(img, split_name, seq_name, frame_number, total, len(objects))
        writer.write(img)

    writer.release()
    return (seq_name, True, f"{total} frames → {out_path.name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="VisDrone-MOT → annotated MP4")
    parser.add_argument("--split",   default="all",
                        choices=["all", "train", "val", "test-dev"])
    parser.add_argument("--fps",     type=int, default=30,
                        help="Output video frame rate (default: 30)")
    parser.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1),
                        help="Parallel workers (default: cpu_count-1)")
    parser.add_argument("--seq",     nargs="+", default=None,
                        help="Only render specific sequence names")
    args = parser.parse_args()

    splits_to_run = list(SPLITS.items()) if args.split == "all" else [(args.split, SPLITS[args.split])]

    tasks = []
    for split_name, split_dir_name in splits_to_run:
        split_dir = MOT_ROOT / split_dir_name
        if not split_dir.exists():
            print(f"  [SKIP] {split_dir} not found")
            continue
        seq_list = sorted(os.listdir(split_dir / "sequences"))
        if args.seq:
            seq_list = [s for s in seq_list if s in args.seq]
        out_dir = OUT_ROOT / split_name
        for seq_name in seq_list:
            tasks.append((split_name, split_dir_name, seq_name, args.fps, str(out_dir)))

    if not tasks:
        print("No tasks to run.")
        return

    print(f"Rendering {len(tasks)} sequences  fps={args.fps}  workers={args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_sequence, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            seq_name, success, msg = fut.result()
            status = "OK " if success else "ERR"
            print(f"  [{i:>3}/{len(tasks)}] {status}  {seq_name}  {msg}")
            if success:
                ok += 1
            else:
                err += 1

    print(f"\nDone: {ok} OK  {err} errors")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
