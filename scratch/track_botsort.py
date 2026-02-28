#!/usr/bin/env python3
"""
BoT-SORT + ReID tracking on a video file via ultralytics built-in tracker.

Usage:
    python3 scratch/track_botsort.py                          # defaults
    python3 scratch/track_botsort.py --source scratch/challenge.mp4
    python3 scratch/track_botsort.py --reid models/reid/yolov8n-cls.pt
    python3 scratch/track_botsort.py --reid none              # IoU-only (no ReID)
"""
import argparse, sys, time, tempfile, os
from pathlib import Path

import cv2
import yaml

_HERE = Path(__file__).resolve().parent

DEFAULT_WEIGHTS = "/home/jetson2/gaurav/yolov8n.pt"
DEFAULT_SOURCE  = str(_HERE / "challenge.mp4")
DEFAULT_REID    = "/home/jetson2/gaurav/models/reid/yolov8n-cls.pt"
REID_DIR        = Path("/home/jetson2/gaurav/models/reid")

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("pip install ultralytics")


def list_reid_models() -> list[str]:
    if REID_DIR.exists():
        return sorted(str(p) for p in REID_DIR.glob("*.pt"))
    return []


def make_tracker_yaml(reid_path: str | None) -> str:
    """Write a temporary botsort yaml and return its path."""
    cfg = {
        "tracker_type": "botsort",
        "track_high_thresh": 0.25,
        "track_low_thresh": 0.10,
        "new_track_thresh": 0.40,  # higher = fewer spurious new tracks from distractors
        "track_buffer": 30,       # keep lost tracks alive for 3s at 50fps
        "match_thresh": 0.7,
        "fuse_score": True,
        "gmc_method": "sparseOptFlow",
        "proximity_thresh": 0.3,    # more permissive spatial match after drift
        "appearance_thresh": 0.5,  # more permissive ReID re-link after long gap
        "with_reid": reid_path is not None,
    }
    if reid_path:
        cfg["model"] = reid_path

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, prefix="botsort_scratch_"
    )
    yaml.dump(cfg, tmp)
    tmp.close()
    return tmp.name


def run(weights: str, source: str, reid_path: str | None,
        imgsz: int, conf: float, save_path: str | None) -> dict:
    """Run BoT-SORT on *source* and return a result dict."""
    tracker_yaml = make_tracker_yaml(reid_path)
    model = YOLO(weights)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        os.unlink(tracker_yaml)
        raise SystemExit(f"Cannot open {source}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 50.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    writer = None
    if save_path:
        writer = cv2.VideoWriter(
            save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H)
        )

    print("=" * 62)
    print(f"  Source    : {source}")
    print(f"  Weights   : {weights}")
    print(f"  ReID      : {reid_path or 'disabled (IoU-only)'}")
    print(f"  Tracker   : {tracker_yaml}")
    print(f"  Save to   : {save_path or 'no'}")
    print("=" * 62)

    id_set: set = set()
    reassignments = 0
    prev_ids: set = set()
    frame_idx = 0
    t0 = time.time()
    t_log = t0

    for result in model.track(
        source=source,
        tracker=tracker_yaml,
        imgsz=imgsz,
        conf=conf,
        stream=True,
        verbose=False,
        persist=True,
    ):
        frame = result.orig_img.copy()
        cur_ids: set = set()

        if result.boxes is not None and result.boxes.id is not None:
            ids  = result.boxes.id.cpu().numpy().astype(int)
            xywh = result.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, xywh):
                cur_ids.add(int(tid))
                id_set.add(int(tid))
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 80), 2)
                label = f"ID {tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x1, max(0, y1-th-6)), (x1+tw+4, y1), (0,255,80), -1)
                cv2.putText(frame, label, (x1+2, max(th, y1-4)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2, cv2.LINE_AA)

        # count ID reassignments: IDs we had before that vanished & new ones appeared
        # (crude proxy for identity switches)
        reassignments += len(cur_ids - prev_ids - id_set) if frame_idx > 0 else 0
        prev_ids = cur_ids

        now = time.time()
        fps_now = (frame_idx + 1) / max(now - t0, 1e-9)
        cv2.putText(frame, f"FPS:{fps_now:.1f}  f{frame_idx+1}/{total}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"IDs:{sorted(cur_ids)}",
                    (8, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 2, cv2.LINE_AA)

        if writer:
            writer.write(frame)

        if now - t_log >= 1.0:
            print(f"  frame={frame_idx+1:4d}/{total} | fps={fps_now:5.1f} | "
                  f"active={sorted(cur_ids)}")
            t_log = now

        frame_idx += 1

    if writer:
        writer.release()
    os.unlink(tracker_yaml)

    elapsed = max(time.time() - t0, 1e-9)
    avg_fps = frame_idx / elapsed
    n_unique_ids = len(id_set)

    print("=" * 62)
    print(f"  Processed  : {frame_idx} frames in {elapsed:.1f}s  ({avg_fps:.1f} fps avg)")
    print(f"  Unique IDs assigned (lower=better): {n_unique_ids}")
    print("=" * 62)

    return {"avg_fps": avg_fps, "unique_ids": n_unique_ids,
            "reid": reid_path or "none", "frames": frame_idx}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--source",  default=DEFAULT_SOURCE)
    p.add_argument("--reid",    default=DEFAULT_REID,
                   help="Path to ReID .pt, or 'none' to disable, or 'all' to benchmark all available")
    p.add_argument("--imgsz",   type=int,   default=640)
    p.add_argument("--conf",    type=float, default=0.25)
    p.add_argument("--save",    default="",
                   help="Output annotated video path. If --reid=all, auto-named per model.")
    return p.parse_args()


def main():
    args = parse_args()

    if args.reid.lower() == "all":
        models = list_reid_models()
        if not models:
            sys.exit(f"No .pt files found in {REID_DIR}")
        print(f"Benchmarking {len(models)} ReID models on {args.source}\n")
        results = []
        for rpath in models:
            name = Path(rpath).stem
            save = str(_HERE / f"botsort_{name}.mp4") if args.save == "" else args.save
            r = run(args.weights, args.source, rpath, args.imgsz, args.conf, save)
            r["name"] = name
            results.append(r)
        print("\n── BENCHMARK SUMMARY ─────────────────────────────────────")
        print(f"{'Model':<25} {'Avg FPS':>8} {'Unique IDs':>12}")
        print("-" * 47)
        for r in sorted(results, key=lambda x: -x["avg_fps"]):
            print(f"  {r['name']:<23} {r['avg_fps']:>8.1f} {r['unique_ids']:>12d}")
        print("─" * 47)

    elif args.reid.lower() == "none":
        save = args.save or str(_HERE / "botsort_nored.mp4")
        run(args.weights, args.source, None, args.imgsz, args.conf, save)
    else:
        save = args.save or str(_HERE / f"botsort_{Path(args.reid).stem}.mp4")
        run(args.weights, args.source, args.reid, args.imgsz, args.conf, save)


if __name__ == "__main__":
    main()
