#!/usr/bin/env python3
"""
Scratch test: YOLOv8n detection + DeepSORT ReID tracking on a video file.
Usage:
    python3 track.py --source ../video_test/13657722_640x640.mp4
    python3 track.py --source ../video_test/13657722_640x640.mp4 --save out.mp4
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# ── detection ───────────────────────────────────────────────────────────────
from ultralytics import YOLO

# ── tracking ─────────────────────────────────────────────────────────────────
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: install deep-sort-realtime first:  pip install deep-sort-realtime", file=sys.stderr)
    raise SystemExit(1)

# ── defaults ─────────────────────────────────────────────────────────────────
_HERE           = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(_HERE.parent / "resources" / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_REID    = "mobilenet"            # DeepSORT built-in embedder (no extra file needed)
DEFAULT_DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"
DEFAULT_SOURCE  = str(_HERE / ".." / "resources" / "video_test" / "13657722_640x640.mp4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO + DeepSORT tracking on a video file")
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS,  help="Detection model (.pt / .engine)")
    p.add_argument("--source",   default=DEFAULT_SOURCE,
                   help="Input video path, or integer camera index (e.g. 0 for webcam)")
    p.add_argument("--imgsz",    type=int,   default=640,  help="Inference image size")
    p.add_argument("--conf",     type=float, default=0.25, help="Detection confidence threshold")
    p.add_argument("--iou",      type=float, default=0.1,  help="NMS IoU threshold (higher = allow more overlapping boxes through YOLO NMS)")
    p.add_argument("--embedder", default=DEFAULT_REID,
                   help="DeepSORT ReID embedder: mobilenet | torchreid | None (IoU-only)")
    p.add_argument("--max-age",  type=int,   default=15,  help="Frames to keep lost tracks (150 = 3s at 50fps)")
    p.add_argument("--n-init",   type=int,   default=5,    help="Hits needed to confirm a track")
    p.add_argument("--nms-max-overlap", type=float, default=1.0,
                   help="DeepSORT internal NMS overlap threshold (1.0 = disabled, e.g. 0.7 to suppress duplicate dets)")
    p.add_argument("--device",   default=DEFAULT_DEVICE, help="Inference device: mps | cpu | cuda")
    p.add_argument("--classes",  nargs="+", type=int, default=[-1],
                   help="Class IDs to track (e.g. 0 2 3). Pass -1 to track all classes.")
    p.add_argument("--save",     default="", help="Output annotated video path (empty = no save)")
    p.add_argument("--show",     action="store_true", help="Display live window")

    # ── annotation arguments ──────────────────────────────────────────────────
    p.add_argument("--hide-det",      action="store_true", help="Hide raw detection boxes")
    p.add_argument("--hide-track",    action="store_true", help="Hide confirmed track boxes")
    p.add_argument("--hide-conf",     action="store_true", help="Hide confidence score on detection boxes")
    p.add_argument("--hide-id",       action="store_true", help="Hide track ID label on track boxes")
    p.add_argument("--hide-class",    action="store_true", help="Hide class name on track boxes")
    p.add_argument("--det-thickness", type=int,   default=1,           help="Detection box line thickness")
    p.add_argument("--track-thickness", type=int, default=2,           help="Track box line thickness")
    p.add_argument("--det-color",     nargs=3, type=int, metavar=("B","G","R"),
                   default=[0, 0, 255],   help="Detection box colour as B G R (default: 0 0 255 = red)")
    p.add_argument("--track-color",   nargs=3, type=int, metavar=("B","G","R"),
                   default=[0, 255, 0],   help="Track box colour as B G R (default: 0 255 0 = green)")
    return p.parse_args()


def make_tracker(args) -> DeepSort:
    use_embedder = None if args.embedder.lower() == "none" else args.embedder
    return DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=0.9,       # more permissive spatial match after drift
        max_cosine_distance=0.4,    # more permissive ReID re-link after long gap
        nn_budget=30,               # larger appearance gallery per track
        nms_max_overlap=args.nms_max_overlap,
        embedder=use_embedder,
        half=False,
        bgr=True,
    )


def main() -> int:
    args = parse_args()

    # ── open video / camera ───────────────────────────────────────────────────
    # Accept integer device index (webcam) or file path
    raw_src = args.source.strip()
    if raw_src.lstrip("-").isdigit():
        cam_idx = int(raw_src)
        cap     = cv2.VideoCapture(cam_idx)
        src     = f"webcam (device {cam_idx})"
        is_live = True
    else:
        src     = str(Path(raw_src).resolve())
        cap     = cv2.VideoCapture(src)
        is_live = False

    if not cap.isOpened():
        print(f"ERROR: Cannot open source: {src}", file=sys.stderr)
        return 1

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
    vid_fps      = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # ── optional writer ───────────────────────────────────────────────────────
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, vid_fps, (width, height))
        if not writer.isOpened():
            cap.release()
            print(f"ERROR: Cannot open output {args.save}", file=sys.stderr)
            return 1

    # ── load models ───────────────────────────────────────────────────────────
    detector = YOLO(args.weights)
    detector.to(args.device)
    is_obb   = detector.task == "obb"
    use_reid = args.embedder.lower() != "none"
    tracker  = make_tracker(args)

    # ── startup summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Source         : {src}")
    print(f"  Resolution     : {width}x{height}  @  {vid_fps:.2f} fps  "
          f"({'live' if is_live else str(total_frames) + ' frames'})")
    print(f"  Detection model: {Path(args.weights).resolve()}")
    print(f"  Model type     : {'OBB (oriented bbox → AABB for tracker)' if is_obb else 'Standard bbox'}")
    print(f"  Tracker        : DeepSORT")
    print(f"  Device         : {args.device}")
    print(f"  Classes        : {'all' if -1 in args.classes else args.classes}")
    print(f"  ReID enabled   : {use_reid}")
    print(f"  ReID embedder  : {args.embedder if use_reid else 'None (IoU-only)'}")
    print(f"  n_init         : {args.n_init}  |  max_age: {args.max_age}")
    print(f"  NMS max overlap: {args.nms_max_overlap} ({'disabled' if args.nms_max_overlap >= 1.0 else 'active'})")
    print(f"  Saving to      : {args.save if args.save else 'no'}")
    print("=" * 60)

    # ── inference loop ────────────────────────────────────────────────────────
    frame_idx   = 0
    t_start     = time.time()
    t_last_log  = t_start
    log_interval = 1.0          # seconds between console prints

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # detection
        result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                  iou=args.iou, device=args.device, verbose=False)[0]

        # build detection list for DeepSORT  →  ([x, y, w, h], conf, class_id)
        # OBB models: use result.obb.xyxy (axis-aligned enclosing rect of the oriented box)
        # Standard models: use result.boxes.xyxy
        track_all = -1 in args.classes
        dets: list = []
        if is_obb:
            src_boxes = result.obb
        else:
            src_boxes = result.boxes

        if src_boxes is not None and len(src_boxes) > 0:
            for box_xyxy, score, cls_id in zip(
                src_boxes.xyxy.cpu().numpy(),
                src_boxes.conf.cpu().numpy(),
                src_boxes.cls.cpu().numpy(),
            ):
                if not track_all and int(cls_id) not in args.classes:
                    continue
                x1, y1, x2, y2 = box_xyxy.tolist()
                w_box = max(4.0, x2 - x1)
                h_box = max(4.0, y2 - y1)
                dets.append(([x1, y1, w_box, h_box], float(score), int(cls_id)))

        # tracking
        if use_reid:
            tracks = tracker.update_tracks(dets, frame=frame)
        else:
            # unit vectors avoid zero-norm division in cosine distance
            unit   = np.ones(128, dtype=np.float32) / np.sqrt(128)
            embeds = [unit] * len(dets)
            tracks = tracker.update_tracks(dets, frame=frame, embeds=embeds)

        confirmed = [t for t in tracks if t.is_confirmed() and
                     (track_all or t.det_class in args.classes)]

        # ── colour scheme ─────────────────────────────────────────────────────
        DET_COLOUR   = tuple(args.det_color)
        TRACK_COLOUR = tuple(args.track_color)

        # annotate frame
        annotated = frame.copy()

        # 1. raw detection boxes
        if not args.hide_det:
            for (box_xywh, score, cls_id) in dets:
                x1 = int(box_xywh[0])
                y1 = int(box_xywh[1])
                x2 = int(box_xywh[0] + box_xywh[2])
                y2 = int(box_xywh[1] + box_xywh[3])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), DET_COLOUR, args.det_thickness)
                if not args.hide_conf:
                    conf_label = f"{score:.2f}"
                    cv2.putText(annotated, conf_label, (x1 + 2, y2 - 4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, DET_COLOUR, 1, cv2.LINE_AA)

        # 2. confirmed track boxes
        if not args.hide_track:
            for track in confirmed:
                ltrb = track.to_ltrb()
                tid  = track.track_id
                x1, y1, x2, y2 = map(int, ltrb)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), TRACK_COLOUR, args.track_thickness)
                parts = []
                if not args.hide_id:
                    parts.append(f"ID {tid}")
                if not args.hide_class and track.det_class is not None:
                    parts.append(detector.names.get(track.det_class, str(track.det_class)))
                label = "  ".join(parts)
                if label:
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, max(0, y1 - th - 6)), (x1 + tw + 4, y1), TRACK_COLOUR, -1)
                    cv2.putText(annotated, label, (x1 + 2, max(th, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

        # FPS overlay + colour legend
        now = time.time()
        elapsed = max(now - t_start, 1e-9)
        fps_now = (frame_idx + 1) / elapsed
        frame_label = f"frame {frame_idx+1}" if is_live else f"frame {frame_idx+1}/{total_frames}"
        cv2.putText(annotated, f"FPS: {fps_now:.1f}  {frame_label}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(annotated, f"Det: {len(dets)}  Tracks: {len(confirmed)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
        if not args.hide_det:
            cv2.putText(annotated, "Detection", (10, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, DET_COLOUR, 2, cv2.LINE_AA)
        if not args.hide_track:
            cv2.putText(annotated, "Track", (110, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, TRACK_COLOUR, 2, cv2.LINE_AA)

        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("YOLO + DeepSORT", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_idx += 1

        # console log every second
        if now - t_last_log >= log_interval:
            track_ids = [t.track_id for t in confirmed]
            total_str = "live" if is_live else str(total_frames)
            print(
                f"frame={frame_idx:4d}/{total_str} | "
                f"fps={fps_now:5.1f} | "
                f"dets={len(dets):2d} | "
                f"confirmed_tracks={len(confirmed)} {track_ids}"
            )
            t_last_log = now

    # ── shutdown ──────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total_time = max(time.time() - t_start, 1e-9)
    print("=" * 60)
    print(f"  Done. Processed {frame_idx} frames in {total_time:.2f}s  "
          f"({frame_idx / total_time:.2f} fps avg)")
    if args.save:
        size_mb = Path(args.save).stat().st_size / 1e6 if Path(args.save).exists() else 0
        print(f"  Saved: {args.save}  ({size_mb:.1f} MB)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
