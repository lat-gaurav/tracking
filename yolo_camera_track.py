#!/usr/bin/env python3
import argparse
import os
import queue
import signal
import sys
import threading
import time

import cv2
from ultralytics import YOLO


DEFAULT_WEIGHTS = "/home/jetson2/gaurav/models/yolon26obb-newmerged-imgsz640-epochs75_fp16.engine"
DEFAULT_TRACKER = "/home/jetson2/gaurav/botsort_reid.yaml"


def read_tracker_config(path: str) -> dict[str, str]:
    cfg: dict[str, str] = {}
    if not os.path.exists(path):
        return cfg

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            cfg[key.strip()] = value.strip()
    return cfg


class AsyncVideoWriter:
    def __init__(self, output_path: str, fps: float, frame_size: tuple[int, int], max_queue: int = 240):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(output_path, fourcc, float(fps), frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Cannot open output video {output_path}")

        self.queue: queue.Queue = queue.Queue(maxsize=max_queue)
        self._stop_token = object()
        self.dropped_frames = 0
        self.written_frames = 0

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is self._stop_token:
                    return
                self.writer.write(item)
                self.written_frames += 1
            finally:
                self.queue.task_done()

    def write(self, frame) -> None:
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.dropped_frames += 1

    def close(self) -> None:
        self.queue.put(self._stop_token)
        self.queue.join()
        self.thread.join(timeout=5.0)
        self.writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Ultralytics YOLO detection + tracking on live camera input."
    )
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Path to YOLO weights/engine")
    parser.add_argument("--device", default="/dev/video0", help="Camera device path")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--tracker", default=DEFAULT_TRACKER, help="Tracker config file")
    parser.add_argument("--show", action="store_true", help="Display annotated live window")
    parser.add_argument("--save", default="", help="Optional output video path (e.g. tracked.mp4)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.device}", file=sys.stderr)
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
    fps = cap.get(cv2.CAP_PROP_FPS) or 60.0
    if fps <= 1.0:
        fps = 60.0

    writer = None
    if args.save:
        try:
            writer = AsyncVideoWriter(args.save, float(fps), (width, height))
        except RuntimeError as exc:
            cap.release()
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1

    model = YOLO(args.weights)

    running = True

    def stop_handler(signum, frame):  # noqa: ARG001
        nonlocal running
        running = False

    signal.signal(signal.SIGINT, stop_handler)
    signal.signal(signal.SIGTERM, stop_handler)

    tracker_cfg = read_tracker_config(args.tracker)
    tracker_type = tracker_cfg.get("tracker_type", "unknown")
    with_reid = tracker_cfg.get("with_reid", "unknown")
    reid_model = tracker_cfg.get("model", "n/a")

    print(f"Camera: {args.device} ({width}x{height} @ ~{fps:.2f} fps)")
    print(f"Detection model: {args.weights}")
    print(f"Tracker config: {args.tracker}")
    print(f"Tracker backend: {tracker_type}")
    print(f"ReID enabled: {with_reid}")
    print(f"ReID model: {reid_model}")
    if args.show:
        print("Press q in the window or Ctrl+C in terminal to stop.")
    else:
        print("Running headless. Press Ctrl+C to stop.")

    processed_frames = 0
    started = time.time()
    last_print = started

    try:
        while running:
            ok, frame = cap.read()
            if not ok:
                print("WARNING: Camera frame read failed, stopping.", file=sys.stderr)
                break

            result = model.track(
                frame,
                persist=True,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                tracker=args.tracker,
                verbose=False,
            )[0]

            annotated = result.plot()
            processed_frames += 1

            now = time.time()
            elapsed = max(now - started, 1e-9)
            infer_fps = processed_frames / elapsed
            cv2.putText(
                annotated,
                f"Infer FPS: {infer_fps:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)

            if args.show:
                cv2.imshow("YOLO Tracking", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if now - last_print >= 1.0:
                detections = int(len(result.boxes)) if result.boxes is not None else 0
                active_trackers = 0
                if result.boxes is not None and result.boxes.id is not None:
                    active_trackers = int(len(result.boxes.id))
                print(
                    "Processed="
                    f"{processed_frames} | infer_fps={infer_fps:.2f} "
                    f"| detections={detections} | active_trackers={active_trackers}"
                )
                last_print = now
    finally:
        cap.release()
        if writer is not None:
            writer.close()
        cv2.destroyAllWindows()

    total = max(time.time() - started, 1e-9)
    print(f"Done. Frames={processed_frames} in {total:.2f}s ({processed_frames / total:.2f} fps)")
    if args.save:
        print(f"Saved: {args.save}")
        print(f"Writer frames={writer.written_frames} | dropped={writer.dropped_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
