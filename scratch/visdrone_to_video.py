"""
visdrone_to_video.py
--------------------
Convert VisDrone2019-DET splits → annotated MP4 videos grouped by sequence.

Annotation format per line:
  <x>,<y>,<w>,<h>,<score>,<category>,<truncation>,<occlusion>

Category IDs:
  0=ignored  1=pedestrian  2=people  3=bicycle  4=car  5=van
  6=truck    7=tricycle     8=awning-tricycle     9=bus  10=motor  11=others

Filename format: <seqID>_<frameOffset>_d_<imgID>.jpg

Usage:
  python3 scratch/visdrone_to_video.py [--split train|val|test-dev|test-challenge|all] [--fps 5]
"""

import argparse, collections
from pathlib import Path
import cv2

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
VISDRONE    = ROOT / "resources" / "visdrone"
OUT_DIR     = ROOT / "resources" / "video_test" / "visdrone_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── category map: id → (name, BGR) ──────────────────────────────────────────
CATEGORIES = {
    0:  ("ignored",          None),               # skip
    1:  ("pedestrian",       (255,  80,  80)),     # blue
    2:  ("people",           (200, 120, 255)),     # violet
    3:  ("bicycle",          (50,  220, 255)),     # yellow
    4:  ("car",              (75,   25, 230)),     # red
    5:  ("van",              (0,   200, 100)),     # green
    6:  ("truck",            (75,  180,  60)),     # lime
    7:  ("tricycle",         (25,  215, 255)),     # gold
    8:  ("awning-tricycle",  (200, 130,   0)),     # teal
    9:  ("bus",              (0,   225, 255)),     # bright yellow
    10: ("motor",            (180,   0, 255)),     # purple
    11: ("others",           (160, 160, 160)),     # grey
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.40
THICK      = 1

SPLIT_DIRS = {
    "train":          "VisDrone2019-DET-train/VisDrone2019-DET-train",
    "val":            "VisDrone2019-DET-val/VisDrone2019-DET-val",
    "test-dev":       "VisDrone2019-DET-test-dev/VisDrone2019-DET-test-dev",
    "test-challenge": "VisDrone2019-DET-test-challenge/VisDrone2019-DET-test-challenge",
}


def classify_sequence(frames: list) -> tuple[bool, float]:
    """Return (is_dense, avg_gap). Dense = consecutive video frames."""
    if len(frames) < 2:
        return False, 0.0
    offsets = [f[0] for f in frames]
    gaps = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]
    avg_gap = sum(gaps) / len(gaps)
    return avg_gap <= 2, avg_gap


def render_sequence(seq_id: str, frames: list, split: str, fps: int) -> str:
    """frames: sorted [(frame_offset, img_path, ann_path_or_None)]"""
    out_path = OUT_DIR / f"{split}_{seq_id}.mp4"
    if out_path.exists():
        return f"[skip]  {split}/{seq_id}"

    is_dense, avg_gap = classify_sequence(frames)

    # Sparse sequences = detection keyframes sampled from long videos.
    # Render as a 1fps slideshow so they don't look like video.
    render_fps = 30.0 if is_dense else 1.0
    mode_tag   = "VIDEO" if is_dense else f"SAMPLED(gap≈{int(avg_gap)}fr)"

    first = cv2.imread(str(frames[0][1]))
    if first is None:
        return f"[ERROR] {split}/{seq_id}: can't read {frames[0][1]}"
    h, w = first.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, render_fps, (w, h))

    for fi, (frame_off, img_path, ann_path) in enumerate(frames):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        obj_counts: dict[str, int] = collections.defaultdict(int)

        if ann_path is not None and ann_path.exists():
            for line in ann_path.read_text().splitlines():
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat_id = int(parts[5])
                trunc  = int(parts[6]) if len(parts) > 6 else 0
                occ    = int(parts[7]) if len(parts) > 7 else 0

                if cat_id == 0:          # ignored region
                    continue
                name, color = CATEGORIES.get(cat_id, ("unknown", (200, 200, 200)))
                if color is None:
                    continue
                obj_counts[name] += 1

                cv2.rectangle(img, (x, y), (x + bw, y + bh), color, 1)

                # label chip: "car" or "car(occ)" if occluded
                suffix = "(occ)" if occ == 2 else ("(par)" if occ == 1 else "")
                disp   = f"{name}{suffix}"
                (tw, th), _ = cv2.getTextSize(disp, FONT, FONT_SCALE, THICK)
                ty = max(y - 2, 10)
                cv2.rectangle(img, (x, ty - th - 1), (x + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(img, disp, (x + 1, ty),
                            FONT, FONT_SCALE, color, THICK, cv2.LINE_AA)

        # HUD
        count_str = "  ".join(f"{k}:{v}" for k, v in sorted(obj_counts.items()))
        hud_lines = [
            f"{split} [{mode_tag}]  seq:{seq_id}  fr:{frame_off:06d}  ({fi+1}/{len(frames)})",
            count_str if count_str else "no annotations",
        ]
        for li, line in enumerate(hud_lines):
            by = 14 + li * 16
            bw2 = len(line) * 8 + 8
            cv2.rectangle(img, (0, by - 12), (bw2, by + 4), (0, 0, 0), -1)
            cv2.putText(img, line, (4, by),
                        FONT, FONT_SCALE, (255, 255, 255), THICK, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    return f"[done]  {split}/{seq_id}  → {out_path.name}  ({len(frames)} frames @ {render_fps:.0f}fps)  [{mode_tag}]"


def process_split(split: str, fps: int):
    rel = SPLIT_DIRS.get(split)
    if rel is None:
        print(f"[warn] Unknown split: {split}")
        return
    split_dir = VISDRONE / rel
    img_dir   = split_dir / "images"
    ann_dir   = split_dir / "annotations"

    if not img_dir.exists():
        print(f"[warn] {img_dir} not found — skipping")
        return

    has_anno = ann_dir.exists()

    # group by seqID
    seq_frames: dict[str, list] = collections.defaultdict(list)
    for img_path in sorted(img_dir.glob("*.jpg")):
        parts      = img_path.stem.split("_")      # [seqID, frameOffset, d, imgID]
        seq_id     = parts[0]
        frame_off  = int(parts[1])
        ann_path   = (ann_dir / (img_path.stem + ".txt")) if has_anno else None
        seq_frames[seq_id].append((frame_off, img_path, ann_path))

    for seq_id in seq_frames:
        seq_frames[seq_id].sort(key=lambda x: x[0])

    total_imgs = sum(len(v) for v in seq_frames.values())
    n_dense  = sum(1 for v in seq_frames.values() if classify_sequence(v)[0])
    n_sparse = len(seq_frames) - n_dense
    print(f"[{split}] {total_imgs} images, {len(seq_frames)} sequences "
          f"({n_dense} dense video, {n_sparse} sparse/sampled)"
          + ("  (no annotations)" if not has_anno else ""))

    for seq_id, frames in sorted(seq_frames.items()):
        print(render_sequence(seq_id, frames, split, fps))


def main():
    ap = argparse.ArgumentParser(description="VisDrone DET → annotated MP4")
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test-dev", "test-challenge", "all"])
    ap.add_argument("--fps",   type=int, default=5,
                    help="Output FPS (images are sparse; 5 is comfortable)")
    args = ap.parse_args()

    splits = list(SPLIT_DIRS.keys()) if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, args.fps)

    print(f"\n[done] Videos → {OUT_DIR}")


if __name__ == "__main__":
    main()
