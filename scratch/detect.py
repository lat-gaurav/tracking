#!/usr/bin/env python3
"""
Detection-only script using YOLO (supports both OBB and standard bbox models).
No tracking – just raw detections drawn every frame.

Usage
-----
    python3 detect.py --source ../video_test/nadir_ped_crossing_crop640.mp4 --show
    python3 detect.py --source 0                          # webcam
    python3 detect.py --source video.mp4 --save out.mp4
    python3 detect.py --source video.mp4 --classes 0 2   # person + car only
    python3 detect.py --source video.mp4 --show --imgsz 1024

Controls (--show)
-----------------
    Q / Esc   quit
    SPACE     pause / resume
    S         save current frame as PNG
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── defaults ──────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(_HERE / ".." / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_SOURCE  = str(_HERE / ".." / "video_test" / "nadir_ped_crossing_crop640.mp4")
DEFAULT_DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"

# colour palette – one per class (cycles if >20 classes)
_PALETTE = [
    (  0, 255,   0), (  0, 200, 255), (255,   0, 128), (255, 180,   0),
    (180,   0, 255), (  0, 128, 255), (255,  80,  80), ( 80, 255, 180),
    (255, 255,   0), (  0, 255, 200), (200, 100, 255), (100, 200,   0),
    (255, 140,  60), ( 60, 140, 255), (200, 255,  60), (255,  60, 200),
    (  0, 180, 180), (180, 180,   0), (255, 100, 100), (100, 255, 100),
]

def cls_colour(cls_id: int):
    return _PALETTE[int(cls_id) % len(_PALETTE)]


# ── arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="YOLO detection-only viewer")
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS,  help="Model path (.pt / .engine / .onnx)")
    p.add_argument("--source",   default=DEFAULT_SOURCE,   help="Video file or camera index")
    p.add_argument("--imgsz",    type=int,   default=640,  help="Inference image size")
    p.add_argument("--conf",     type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou",      type=float, default=0.45, help="NMS IoU threshold")
    p.add_argument("--classes",  nargs="+",  type=int,     help="Filter to these class IDs (default: all)")
    p.add_argument("--device",   default=DEFAULT_DEVICE,   help="mps | cpu | cuda")
    p.add_argument("--save",     default="",               help="Output video path")
    p.add_argument("--show",     action="store_true",      help="Display live window")
    p.add_argument("--hide-labels",  action="store_true",  help="Draw boxes without text labels")
    p.add_argument("--hide-conf",    action="store_true",  help="Omit confidence score from label")
    p.add_argument("--thickness",    type=int, default=2,  help="Box line thickness")
    return p.parse_args()


# ── drawing ───────────────────────────────────────────────────────────────────
def draw_box(img, x1, y1, x2, y2, label, color, thickness=2, hide_label=False):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if not hide_label and label:
        scale = 0.55
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th + bl, y1)
        cv2.rectangle(img, (x1, ty - th - bl - 1), (x1 + tw + 4, ty + 1), color, -1)
        luma = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        fg = (0, 0, 0) if luma > 128 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


def draw_obb(img, pts, label, color, thickness=2, hide_label=False):
    """Draw an oriented bounding box from 4 corner points (N,2)."""
    pts_i = pts.astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(img, [pts_i], True, color, thickness)
    if not hide_label and label:
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        scale = 0.55
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = max(th + bl, y1)
        cv2.rectangle(img, (x1, ty - th - bl - 1), (x1 + tw + 4, ty + 1), color, -1)
        luma = 0.299 * color[2] + 0.587 * color[1] + 0.114 * color[0]
        fg = (0, 0, 0) if luma > 128 else (255, 255, 255)
        cv2.putText(img, label, (x1 + 2, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, fg, 1, cv2.LINE_AA)


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    # open source
    raw = args.source.strip()
    if raw.lstrip("-").isdigit():
        cap = cv2.VideoCapture(int(raw))
        src_label = f"webcam {raw}"
        is_live = True
    else:
        cap = cv2.VideoCapture(str(Path(raw).resolve()))
        src_label = Path(raw).name
        is_live = False

    if not cap.isOpened():
        print(f"ERROR: cannot open {raw}", file=sys.stderr)
        return 1

    fps_in   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H        = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (W, H))
        if not writer.isOpened():
            cap.release()
            print(f"ERROR: cannot open output {args.save}", file=sys.stderr)
            return 1

    # load model
    model   = YOLO(args.weights)
    model.to(args.device)
    is_obb  = model.task == "obb"
    names   = model.names          # {id: name}

    filter_cls = set(args.classes) if args.classes else None

    print("=" * 60)
    print(f"  Source   : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")
    print(f"  Model    : {Path(args.weights).name}")
    print(f"  Type     : {'OBB (oriented bounding box)' if is_obb else 'Standard bbox'}")
    print(f"  Device   : {args.device}")
    print(f"  Classes  : {list(filter_cls) if filter_cls else 'all'}")
    print(f"  Conf     : {args.conf}   IoU: {args.iou}")
    print("=" * 60)

    if args.show:
        cv2.namedWindow("Detect", cv2.WINDOW_NORMAL)

    frame_idx   = 0
    t0          = time.time()
    t_last_log  = t0
    paused      = False
    snap_count  = 0
    annotated   = None

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = False
            if key == ord("s") and annotated is not None:
                fname = f"snap_{snap_count:04d}.png"
                cv2.imwrite(fname, annotated)
                print(f"  Saved snapshot → {fname}")
                snap_count += 1
            continue

        # ── inference ────────────────────────────────────────────────────────
        result = model.predict(frame, imgsz=args.imgsz, conf=args.conf,
                               iou=args.iou, device=args.device, verbose=False)[0]

        # ── collect detections ───────────────────────────────────────────────
        annotated = frame.copy()
        n_dets = 0

        if is_obb:
            boxes = result.obb
        else:
            boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            confs   = boxes.conf.cpu().numpy()
            cls_ids = boxes.cls.cpu().numpy().astype(int)

            if is_obb:
                # corners: shape (N, 4, 2)
                corners = boxes.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            else:
                xyxys = boxes.xyxy.cpu().numpy()

            for i, (conf, cls_id) in enumerate(zip(confs, cls_ids)):
                if filter_cls and cls_id not in filter_cls:
                    continue
                n_dets += 1
                color = cls_colour(cls_id)
                cls_name = names.get(cls_id, str(cls_id))
                if args.hide_labels:
                    label = ""
                elif args.hide_conf:
                    label = cls_name
                else:
                    label = f"{cls_name} {conf:.2f}"

                if is_obb:
                    draw_obb(annotated, corners[i], label, color,
                             args.thickness, args.hide_labels)
                else:
                    x1, y1, x2, y2 = map(int, xyxys[i])
                    draw_box(annotated, x1, y1, x2, y2, label, color,
                             args.thickness, args.hide_labels)

        # ── HUD ──────────────────────────────────────────────────────────────
        now     = time.time()
        elapsed = max(now - t0, 1e-9)
        fps_now = (frame_idx + 1) / elapsed
        frame_str = f"frame {frame_idx + 1}" if is_live else f"frame {frame_idx + 1}/{n_frames}"
        cv2.putText(annotated,
                    f"FPS {fps_now:.1f}  |  {frame_str}  |  dets {n_dets}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, "Q=quit  SPACE=pause  S=snapshot",
                    (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        if now - t_last_log >= 1.0:
            print(f"frame={frame_idx+1:5d}/{n_frames if not is_live else 'live'} | "
                  f"fps={fps_now:5.1f} | dets={n_dets:3d}")
            t_last_log = now

        # ── output ───────────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)

        if args.show:
            cv2.imshow("Detect", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = True
                print("  [paused]")
            if key == ord("s"):
                fname = f"snap_{snap_count:04d}.png"
                cv2.imwrite(fname, annotated)
                print(f"  Saved snapshot → {fname}")
                snap_count += 1

        frame_idx += 1

    # ── shutdown ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total = max(time.time() - t0, 1e-9)
    print("=" * 60)
    print(f"  Done. {frame_idx} frames in {total:.1f}s  ({frame_idx / total:.1f} fps avg)")
    if args.save and Path(args.save).exists():
        print(f"  Saved → {args.save}  ({Path(args.save).stat().st_size / 1e6:.1f} MB)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
