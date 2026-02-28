#!/usr/bin/env python3
"""
Click-to-Track: click on any object in the video to lock onto it.
The tracker runs on ALL detections, but only the clicked object is highlighted.

Controls
--------
  Left-click      – select the object under the cursor
  R               – reset / re-select a new object
  SPACE           – pause / resume
  Q / Esc         – quit

Usage
-----
    python3 click_to_track.py --source ../video_test/nadir_crosswalk_ped.mp4 --show
    python3 click_to_track.py --source 0                        # webcam
    python3 click_to_track.py --source video.mp4 --save out.mp4
"""
import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except ImportError:
    print("ERROR: pip install deep-sort-realtime", file=sys.stderr)
    raise SystemExit(1)

# ── defaults ──────────────────────────────────────────────────────────────────
_HERE          = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = str(_HERE / ".." / "models" / "yolov26nobbnew_merged_1024.pt")
DEFAULT_SOURCE  = str(_HERE / ".." / "video_test" / "nadir_crosswalk_ped.mp4")
DEFAULT_DEVICE  = "mps" if torch.backends.mps.is_available() else "cpu"
TRAIL_LEN       = 60          # number of past centre-points to draw as trail
LOST_PATIENCE   = 45          # frames to wait after a track disappears before showing "Lost"

# colours (BGR)
COL_DIM_BOX      = (80,  80,  80)    # grey  – background tracks
COL_DIM_DET      = (40,  40, 100)    # dark  – background detections
COL_LOCKED       = (0,  230, 255)    # cyan  – the locked track
COL_TRAIL        = (0,  165, 255)    # orange trail dots
COL_LOST         = (0,   0, 255)     # red   – "Track Lost" text
COL_HINT         = (255, 255,  0)    # yellow – instructions


# ── arg parsing ───────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--weights",  default=DEFAULT_WEIGHTS)
    p.add_argument("--source",   default=DEFAULT_SOURCE)
    p.add_argument("--imgsz",    type=int,   default=640)
    p.add_argument("--conf",     type=float, default=0.25)
    p.add_argument("--iou",      type=float, default=0.45)
    p.add_argument("--embedder", default="none",
                   help="DeepSORT ReID embedder: mobilenet | torchreid | none")
    p.add_argument("--max-age",  type=int,   default=30)
    p.add_argument("--n-init",   type=int,   default=3)
    p.add_argument("--device",   default=DEFAULT_DEVICE)
    p.add_argument("--save",     default="", help="Output video path")
    p.add_argument("--show",     action="store_true")
    return p.parse_args()


# ── tracker factory ───────────────────────────────────────────────────────────
def make_tracker(args):
    use_embedder = None if args.embedder.lower() == "none" else args.embedder
    return DeepSort(
        max_age=args.max_age,
        n_init=args.n_init,
        max_iou_distance=0.9,
        max_cosine_distance=0.4,
        nn_budget=30,
        embedder=use_embedder,
        half=False,
        bgr=True,
    )


# ── click state (shared with mouse callback) ──────────────────────────────────
class ClickState:
    def __init__(self):
        self.pending_click = None   # (x, y) set by mouse callback, consumed by main loop
        self.locked_id     = None   # currently tracked track_id
        self.trail         = deque(maxlen=TRAIL_LEN)   # (cx, cy) history
        self.lost_frames   = 0      # frames since locked track was last seen


def mouse_cb(event, x, y, flags, state: ClickState):
    if event == cv2.EVENT_LBUTTONDOWN:
        state.pending_click = (x, y)


# ── geometry helpers ──────────────────────────────────────────────────────────
def ltrb_to_xywh(l, t, r, b):
    return l, t, r - l, b - t


def point_in_box(px, py, l, t, r, b):
    return l <= px <= r and t <= py <= b


def box_centre(l, t, r, b):
    return (l + r) / 2, (t + b) / 2


def dist2(ax, ay, bx, by):
    return (ax - bx) ** 2 + (ay - by) ** 2


def find_track_at(click_x, click_y, confirmed_tracks):
    """Return the track_id of the confirmed track whose box contains the click,
    or the nearest one by centroid distance (fallback)."""
    # 1. exact hit
    for t in confirmed_tracks:
        l, top, r, b = map(int, t.to_ltrb())
        if point_in_box(click_x, click_y, l, top, r, b):
            return t.track_id
    # 2. nearest centroid
    if not confirmed_tracks:
        return None
    best = min(
        confirmed_tracks,
        key=lambda t: dist2(click_x, click_y, *box_centre(*map(int, t.to_ltrb())))
    )
    cx, cy = box_centre(*map(int, best.to_ltrb()))
    if dist2(click_x, click_y, cx, cy) < 100 ** 2:   # within 100 px
        return best.track_id
    return None


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_dashed_rect(img, pt1, pt2, color, thickness=1, dash=8):
    """Draw a dashed rectangle."""
    x1, y1 = pt1
    x2, y2 = pt2
    pts = [
        ((x1, y1), (x2, y1)),
        ((x2, y1), (x2, y2)),
        ((x2, y2), (x1, y2)),
        ((x1, y2), (x1, y1)),
    ]
    for (ax, ay), (bx, by) in pts:
        length = int(np.hypot(bx - ax, by - ay))
        if length == 0:
            continue
        dx, dy = (bx - ax) / length, (by - ay) / length
        for i in range(0, length, dash * 2):
            sx = int(ax + dx * i)
            sy = int(ay + dy * i)
            ex = int(ax + dx * min(i + dash, length))
            ey = int(ay + dy * min(i + dash, length))
            cv2.line(img, (sx, sy), (ex, ey), color, thickness)


def put_label(img, text, x, y, fg=(0, 0, 0), bg=COL_LOCKED, scale=0.6, thick=2):
    (tw, th), bl = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    cv2.rectangle(img, (x, max(0, y - th - bl)), (x + tw + 4, y + bl), bg, -1)
    cv2.putText(img, text, (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


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

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, fps_in, (W, H))

    # models
    detector = YOLO(args.weights)
    detector.to(args.device)
    is_obb  = detector.task == "obb"
    tracker = make_tracker(args)
    use_reid = args.embedder.lower() != "none"

    print("=" * 60)
    print(f"  Source  : {src_label}  ({W}x{H} @ {fps_in:.1f} fps)")
    print(f"  Model   : {Path(args.weights).name}  ({'OBB' if is_obb else 'bbox'})")
    print(f"  Device  : {args.device}")
    print(f"  Click the object you want to track.  Press R to re-select.")
    print("=" * 60)

    # click state
    state = ClickState()

    if args.show:
        cv2.namedWindow("Click-to-Track", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Click-to-Track", mouse_cb, state)

    frame_idx = 0
    t0 = time.time()
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break
        else:
            # keep showing last annotated frame while paused
            key = cv2.waitKey(50) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord(" "):
                paused = False
            if key == ord("r"):
                state.locked_id = None
                state.trail.clear()
                state.lost_frames = 0
                print("Selection reset.")
            continue

        # ── detection ────────────────────────────────────────────────────────
        result = detector.predict(frame, imgsz=args.imgsz, conf=args.conf,
                                  iou=args.iou, device=args.device, verbose=False)[0]

        src_boxes = result.obb if is_obb else result.boxes
        dets = []
        if src_boxes is not None and len(src_boxes) > 0:
            for xyxy, score, cls_id in zip(
                src_boxes.xyxy.cpu().numpy(),
                src_boxes.conf.cpu().numpy(),
                src_boxes.cls.cpu().numpy(),
            ):
                x1, y1, x2, y2 = xyxy.tolist()
                w = max(4.0, x2 - x1)
                h = max(4.0, y2 - y1)
                dets.append(([x1, y1, w, h], float(score), int(cls_id)))

        # ── tracking ─────────────────────────────────────────────────────────
        if use_reid:
            tracks = tracker.update_tracks(dets, frame=frame)
        else:
            unit   = np.ones(128, dtype=np.float32) / np.sqrt(128)
            embeds = [unit] * len(dets)
            tracks = tracker.update_tracks(dets, frame=frame, embeds=embeds)

        confirmed = [t for t in tracks if t.is_confirmed()]

        # ── handle click: assign locked_id ───────────────────────────────────
        if state.pending_click is not None:
            cx, cy = state.pending_click
            state.pending_click = None
            tid = find_track_at(cx, cy, confirmed)
            if tid is not None:
                state.locked_id   = tid
                state.trail.clear()
                state.lost_frames = 0
                print(f"  → Locked onto track ID {tid}")
            else:
                print(f"  No confirmed track near ({cx},{cy}) – try again.")

        # ── update trail & lost counter ───────────────────────────────────────
        locked_track = None
        if state.locked_id is not None:
            for t in confirmed:
                if t.track_id == state.locked_id:
                    locked_track = t
                    break
            if locked_track is not None:
                l, top, r, b = map(int, locked_track.to_ltrb())
                state.trail.append(
                    (int((l + r) / 2), int((top + b) / 2))
                )
                state.lost_frames = 0
            else:
                state.lost_frames += 1

        # ── annotate ─────────────────────────────────────────────────────────
        annotated = frame.copy()

        # 1. dim background detections
        for (box_xywh, score, _) in dets:
            x1 = int(box_xywh[0]);  y1 = int(box_xywh[1])
            x2 = int(x1 + box_xywh[2]); y2 = int(y1 + box_xywh[3])
            cv2.rectangle(annotated, (x1, y1), (x2, y2), COL_DIM_DET, 1)

        # 2. dim background tracks
        for t in confirmed:
            if t.track_id == state.locked_id:
                continue
            l, top, r, b = map(int, t.to_ltrb())
            cv2.rectangle(annotated, (l, top), (r, b), COL_DIM_BOX, 1)
            cv2.putText(annotated, f"ID{t.track_id}", (l + 2, top + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COL_DIM_BOX, 1, cv2.LINE_AA)

        # 3. trail
        for i, (px, py) in enumerate(state.trail):
            alpha = int(80 + 175 * i / max(1, len(state.trail) - 1))
            r_dot = max(2, i // 6)
            cv2.circle(annotated, (px, py), r_dot, COL_TRAIL, -1)

        # 4. locked track (or "lost" overlay)
        if state.locked_id is not None:
            if locked_track is not None:
                l, top, r, b = map(int, locked_track.to_ltrb())
                # filled semi-transparent highlight
                overlay = annotated.copy()
                cv2.rectangle(overlay, (l, top), (r, b), COL_LOCKED, -1)
                cv2.addWeighted(overlay, 0.15, annotated, 0.85, 0, annotated)
                # bold border
                cv2.rectangle(annotated, (l, top), (r, b), COL_LOCKED, 3)
                # corner accents
                corner = 20
                for (cx2, cy2), (dx, dy) in [
                    ((l, top), (1, 1)), ((r, top), (-1, 1)),
                    ((r, b), (-1, -1)), ((l, b), (1, -1))
                ]:
                    cv2.line(annotated, (cx2, cy2), (cx2 + dx * corner, cy2), COL_LOCKED, 3)
                    cv2.line(annotated, (cx2, cy2), (cx2, cy2 + dy * corner), COL_LOCKED, 3)
                # label
                cls_name = detector.names.get(locked_track.det_class, f"cls{locked_track.det_class}") \
                           if locked_track.det_class is not None else "obj"
                label = f"ID {state.locked_id} | {cls_name}"
                put_label(annotated, label, l, max(14, top - 4),
                          fg=(0, 0, 0), bg=COL_LOCKED, scale=0.65, thick=2)
            elif state.lost_frames < LOST_PATIENCE:
                # draw last known trail end
                if state.trail:
                    lx, ly = state.trail[-1]
                    draw_dashed_rect(annotated, (lx - 40, ly - 40), (lx + 40, ly + 40),
                                     COL_LOST, 2)
                cv2.putText(annotated,
                            f"Track {state.locked_id} lost ({state.lost_frames}/{LOST_PATIENCE})...",
                            (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, COL_LOST, 2, cv2.LINE_AA)
            else:
                # give up
                cv2.putText(annotated, f"Track {state.locked_id} permanently lost. Press R to re-select.",
                            (10, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_LOST, 2, cv2.LINE_AA)

        # 5. HUD
        elapsed = max(time.time() - t0, 1e-9)
        fps_now = (frame_idx + 1) / elapsed
        frame_str = f"frame {frame_idx + 1}" if is_live else f"frame {frame_idx + 1}/{n_frames}"
        cv2.putText(annotated, f"FPS {fps_now:.1f}  {frame_str}  dets {len(dets)}  tracks {len(confirmed)}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)

        # selection hint
        if state.locked_id is None:
            hint = "Click on an object to track it"
        else:
            hint = "R = re-select   |   SPACE = pause   |   Q = quit"
        cv2.putText(annotated, hint, (10, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COL_HINT, 1, cv2.LINE_AA)

        # crosshair while nothing selected
        if state.locked_id is None:
            cx2, cy2 = W // 2, H // 2
            cv2.line(annotated, (cx2 - 20, cy2), (cx2 + 20, cy2), COL_HINT, 1)
            cv2.line(annotated, (cx2, cy2 - 20), (cx2, cy2 + 20), COL_HINT, 1)

        # ── output ───────────────────────────────────────────────────────────
        if writer:
            writer.write(annotated)
        if args.show:
            cv2.imshow("Click-to-Track", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("r"):
                state.locked_id = None
                state.trail.clear()
                state.lost_frames = 0
                print("Selection reset.")
            if key == ord(" "):
                paused = True

        frame_idx += 1

    # ── cleanup ───────────────────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    total = max(time.time() - t0, 1e-9)
    print("=" * 60)
    print(f"  Done. {frame_idx} frames in {total:.1f}s ({frame_idx / total:.1f} fps avg)")
    if args.save and Path(args.save).exists():
        print(f"  Saved → {args.save}  ({Path(args.save).stat().st_size / 1e6:.1f} MB)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
