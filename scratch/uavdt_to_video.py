"""
uavdt_to_video.py
-----------------
Convert UAVDT-sample (Supervisely format) → annotated MP4 videos.
Groups scattered frames by sequence ID, sorts by frame number,
draws multi-class bboxes and a HUD on every frame.

Usage:
  python3 scratch/uavdt_to_video.py [--split train|test|all] [--fps 5]
"""

import json, argparse, collections
from pathlib import Path
import cv2

# ─── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
UAVDT_DIR  = ROOT / "resources" / "uavdt_sample"
OUT_DIR    = ROOT / "resources" / "video_test" / "uavdt_annotated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── class colours & names ────────────────────────────────────────────────────
# parsed from meta.json: #hex → BGR
CLASS_INFO = {
    6544596: ("bus",     (25,  225, 255)),   # #FFE119 → BGR
    6544594: ("car",     (75,  25,  230)),   # #E6194B → BGR
    6544595: ("truck",   (75,  180,  60)),   # #3CB44B → BGR
    6544597: ("vehicle", (200, 130,   0)),   # #0082C8 → BGR
}
FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.42
THICK      = 1


def hex_to_bgr(h: str):
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def load_meta(uavdt_dir: Path) -> dict:
    """Return classId → (title, BGR_color)"""
    meta = json.loads((uavdt_dir / "meta.json").read_text())
    return {c["id"]: (c["title"], hex_to_bgr(c["color"])) for c in meta["classes"]}


def render_sequence(seq_id: str, frames: list[tuple], split: str,
                    class_map: dict, fps: int) -> str:
    """frames: sorted list of (frame_no, img_path, ann_path)"""
    out_path = OUT_DIR / f"{split}_{seq_id}.mp4"
    if out_path.exists():
        return f"[skip]  {split}/{seq_id}  (already exists)"

    # probe size from first image
    first_img = cv2.imread(str(frames[0][1]))
    if first_img is None:
        return f"[ERROR] {split}/{seq_id}: can't read {frames[0][1]}"
    h, w = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (w, h))

    for fi, (frame_no, img_path, ann_path) in enumerate(frames):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # load annotation
        obj_counts: dict[str, int] = collections.defaultdict(int)
        if ann_path.exists():
            ann = json.loads(ann_path.read_text())
            for obj in ann.get("objects", []):
                cid   = obj["classId"]
                label, color = class_map.get(cid, (obj.get("classTitle", "?"), (200, 200, 200)))
                label = obj.get("classTitle", label)   # use classTitle directly if present
                obj_counts[label] += 1

                # extract target id from tags
                tid = None
                for tag in obj.get("tags", []):
                    if tag.get("name") == "target id":
                        tid = tag.get("value")
                        break

                ext = obj["points"]["exterior"]      # [[x1,y1],[x2,y2]]
                x1, y1 = int(ext[0][0]), int(ext[0][1])
                x2, y2 = int(ext[1][0]), int(ext[1][1])
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # label: "car#7" or just "car" if no id
                disp = f"{label}#{tid}" if tid is not None else label
                # background chip for readability
                (tw, th), _ = cv2.getTextSize(disp, FONT, FONT_SCALE, THICK)
                ty = max(y1 - 3, 10)
                cv2.rectangle(img, (x1, ty - th - 1), (x1 + tw + 2, ty + 2), (0, 0, 0), -1)
                cv2.putText(img, disp, (x1 + 1, ty),
                            FONT, FONT_SCALE, color, THICK, cv2.LINE_AA)

        # HUD — sequence name + frame number + object count per class
        count_str = "  ".join(f"{k}:{v}" for k, v in sorted(obj_counts.items()))
        hud_lines = [
            f"{split.upper()}  seq:{seq_id}  fr:{frame_no:06d}  ({fi+1}/{len(frames)})",
            count_str if count_str else "no objects",
        ]
        for li, line in enumerate(hud_lines):
            by = 14 + li * 16
            bw = len(line) * 8 + 8
            cv2.rectangle(img, (0, by - 12), (bw, by + 4), (0, 0, 0), -1)
            cv2.putText(img, line, (4, by),
                        FONT, FONT_SCALE, (255, 255, 255), THICK, cv2.LINE_AA)

        writer.write(img)

    writer.release()
    return f"[done]  {split}/{seq_id}  → {out_path.name}  ({len(frames)} frames)"


def process_split(split: str, class_map: dict, fps: int):
    split_dir = UAVDT_DIR / split
    img_dir   = split_dir / "img"
    ann_dir   = split_dir / "ann"

    if not img_dir.exists():
        print(f"[warn] {img_dir} not found — skipping")
        return

    # group by sequence
    seq_frames: dict[str, list] = collections.defaultdict(list)
    for img_path in sorted(img_dir.glob("*.jpg")):
        name     = img_path.stem                          # e.g. M0101_img000016
        seq_id   = name.split("_")[0]                    # M0101
        frame_no = int(name.split("img")[-1])            # 16
        ann_path = ann_dir / (img_path.name + ".json")
        seq_frames[seq_id].append((frame_no, img_path, ann_path))

    # sort each sequence by frame number
    for seq_id in seq_frames:
        seq_frames[seq_id].sort(key=lambda x: x[0])

    print(f"[{split}] {sum(len(v) for v in seq_frames.values())} images "
          f"across {len(seq_frames)} sequences → {fps} fps")

    for seq_id, frames in sorted(seq_frames.items()):
        print(render_sequence(seq_id, frames, split, class_map, fps))


def main():
    ap = argparse.ArgumentParser(description="UAVDT sample → annotated MP4")
    ap.add_argument("--split",   default="all", choices=["train", "test", "all"])
    ap.add_argument("--fps",     type=int, default=5,
                    help="Output FPS (images are sparse; 5 is comfortable)")
    args = ap.parse_args()

    class_map = load_meta(UAVDT_DIR)
    print(f"[meta] {len(class_map)} classes: {[v[0] for v in class_map.values()]}")

    splits = ["train", "test"] if args.split == "all" else [args.split]
    for split in splits:
        process_split(split, class_map, args.fps)

    print(f"\n[done] Videos → {OUT_DIR}")


if __name__ == "__main__":
    main()
