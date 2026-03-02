#!/usr/bin/env python3
"""
M3OT → annotated MP4 visualizer
Renders each RGB and IR sequence as an MP4 with MOT-format bounding boxes
labeled #<track_id>.

Dataset layout:
  resources/M3OT/
    {1,2}/                        ← scene
      {rgb,ir}/
        {train,val,test}/
          {seq_name}/
            img1/                 ← 000001.PNG …
            gt/gt.txt             ← MOT CSV: frame,id,x,y,w,h,conf,class,vis
            seqinfo.ini
    Annotations/                  ← COCO JSON (not used here)

MOT gt.txt columns (1-indexed):
  frame, track_id, x, y, w, h, confidence, class, visibility
  (x,y) = top-left corner  (float), w/h = float

Usage:
  python3 scratch/m3ot_to_video.py                      # all
  python3 scratch/m3ot_to_video.py --modality rgb
  python3 scratch/m3ot_to_video.py --modality ir
  python3 scratch/m3ot_to_video.py --scene 1
  python3 scratch/m3ot_to_video.py --split train
  python3 scratch/m3ot_to_video.py --fps 10 --workers 4
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
ROOT     = Path(__file__).resolve().parent.parent
M3OT_ROOT = ROOT / "resources" / "M3OT"
OUT_ROOT  = ROOT / "resources" / "video_test" / "m3ot_annotated"

SCENES     = ["1", "2"]
MODALITIES = ["rgb", "ir"]
SPLITS     = ["train", "val", "test"]

# ---------------------------------------------------------------------------

def _track_color(track_id: int) -> tuple:
    """Deterministic distinct BGR color per track_id via golden-ratio hue."""
    h = (track_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))


def _load_gt(gt_path: Path) -> dict[int, list]:
    """Return frame (1-based) → list of (track_id, x, y, w, h)."""
    frames: dict[int, list] = defaultdict(list)
    if not gt_path.exists():
        return frames
    with gt_path.open() as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            frame  = int(row[0])
            tid    = int(row[1])
            x, y   = float(row[2]), float(row[3])
            w, h   = float(row[4]), float(row[5])
            if w > 0 and h > 0:
                frames[frame].append((tid, int(x), int(y), int(w), int(h)))
    return frames


def _draw_label(img, text: str, x: int, y: int, color: tuple):
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.32
    thickness  = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    lx = max(x, 0)
    ly = max(y - th - baseline - 1, 0)
    cv2.rectangle(img, (lx, ly), (lx + tw + 2, ly + th + baseline + 1), color, -1)
    brightness = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
    text_color = (0, 0, 0) if brightness > 140 else (255, 255, 255)
    cv2.putText(img, text, (lx + 1, ly + th), font, font_scale,
                text_color, thickness, cv2.LINE_AA)


def _hud(img, scene: str, modality: str, split: str, seq: str,
         frame_no: int, total: int, n_objects: int):
    h, w = img.shape[:2]
    bar_h = 26
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    # IR = yellow tint indicator
    mod_tag = f"[{modality.upper()}]"
    text = (f"[M3OT s{scene}/{split}] {mod_tag}  {seq}"
            f"   frame {frame_no:05d}/{total:05d}   objects: {n_objects}")
    cv2.putText(img, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (220, 220, 220), 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------

def render_sequence(task):
    scene, modality, split, seq_name, fps_override, out_dir_str = task
    seq_dir  = M3OT_ROOT / scene / modality / split / seq_name
    img_dir  = seq_dir / "img1"
    gt_path  = seq_dir / "gt" / "gt.txt"
    out_dir  = Path(out_dir_str)
    out_dir.mkdir(parents=True, exist_ok=True)

    # read seqinfo for native fps
    seqinfo  = seq_dir / "seqinfo.ini"
    native_fps = fps_override
    seq_len    = 0
    if seqinfo.exists():
        for line in seqinfo.read_text().splitlines():
            if line.lower().startswith("framerate"):
                try:
                    native_fps = fps_override or int(line.split("=")[1])
                except Exception:
                    pass
            if line.lower().startswith("seqlength"):
                try:
                    seq_len = int(line.split("=")[1])
                except Exception:
                    pass

    ann = _load_gt(gt_path)

    img_ext  = ".PNG"
    frame_paths = sorted(img_dir.glob(f"*{img_ext}"),
                         key=lambda p: int(p.stem))
    if not frame_paths:
        frame_paths = sorted(img_dir.glob("*.jpg"),
                             key=lambda p: int(p.stem))
    if not frame_paths:
        return (seq_name, False, "no images found")

    total    = len(frame_paths)
    tag      = f"s{scene}_{modality}_{split}_{seq_name}"
    out_path = out_dir / f"{tag}.mp4"

    sample = cv2.imread(str(frame_paths[0]))
    if sample is None:
        return (seq_name, False, "cannot read first frame")

    # IR images are grayscale PNGs — read and convert to BGR for colour overlay
    fh, fw = sample.shape[:2]
    if len(sample.shape) == 2 or sample.shape[2] == 1:
        sample = cv2.cvtColor(sample, cv2.COLOR_GRAY2BGR)

    fps_final = native_fps if native_fps else 10
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_final, (fw, fh))

    for frame_idx, fp in enumerate(frame_paths):
        frame_no = frame_idx + 1   # MOT is 1-indexed
        raw = cv2.imread(str(fp))
        if raw is None:
            continue
        img = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if (
            len(raw.shape) == 2 or raw.shape[2] == 1) else raw

        objects = ann.get(frame_no, [])
        for (tid, x, y, w, h) in objects:
            color = _track_color(tid)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            _draw_label(img, f"#{tid}", x, y, color)

        _hud(img, scene, modality, split, seq_name, frame_no, total, len(objects))
        writer.write(img)

    writer.release()
    return (seq_name, True, f"{total} fr  fps={fps_final} → {out_path.name}")


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="M3OT → annotated MP4")
    parser.add_argument("--scene",    default="all",
                        choices=["all", "1", "2"])
    parser.add_argument("--modality", default="all",
                        choices=["all", "rgb", "ir"])
    parser.add_argument("--split",    default="all",
                        choices=["all", "train", "val", "test"])
    parser.add_argument("--fps",      type=int, default=None,
                        help="Override fps (default: read from seqinfo.ini)")
    parser.add_argument("--workers",  type=int, default=max(1, os.cpu_count() - 1))
    parser.add_argument("--seq",      nargs="+", default=None,
                        help="Only render specific sequence names")
    args = parser.parse_args()

    scenes     = SCENES     if args.scene    == "all" else [args.scene]
    modalities = MODALITIES if args.modality == "all" else [args.modality]
    splits     = SPLITS     if args.split    == "all" else [args.split]

    tasks = []
    for scene in scenes:
        for modality in modalities:
            for split in splits:
                split_dir = M3OT_ROOT / scene / modality / split
                if not split_dir.exists():
                    continue
                out_dir = OUT_ROOT / f"s{scene}" / modality
                for seq_dir in sorted(split_dir.iterdir()):
                    if not seq_dir.is_dir():
                        continue
                    if args.seq and seq_dir.name not in args.seq:
                        continue
                    tasks.append((scene, modality, split, seq_dir.name,
                                  args.fps, str(out_dir)))

    if not tasks:
        print("No tasks found — check M3OT folder structure.")
        return

    print(f"Rendering {len(tasks)} sequences  workers={args.workers}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    ok = err = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(render_sequence, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            seq_name, success, msg = fut.result()
            status = "OK " if success else "ERR"
            print(f"  [{i:>3}/{len(tasks)}] {status}  {seq_name:<12}  {msg}")
            ok += success
            err += not success

    print(f"\nDone: {ok} OK  {err} errors")
    print(f"Output: {OUT_ROOT}")


if __name__ == "__main__":
    main()
