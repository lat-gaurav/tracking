#!/usr/bin/env python3
import signal
import subprocess
import sys


DEVICE = "/dev/video0"
OUTPUT = "recording.yuy2"
WIDTH = 1920
HEIGHT = 1080
FPS_NUM = 60000
FPS_DEN = 1001


def run_pipeline(cmd: list[str]) -> int:
    proc = subprocess.Popen(cmd)

    def handle_stop(signum, frame):  # noqa: ARG001
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    return proc.wait()


def raw_pipeline() -> list[str]:
    return [
        "gst-launch-1.0",
        "-e",
        "v4l2src",
        f"device={DEVICE}",
        "io-mode=2",
        "!",
        "video/x-raw,format=YUY2," f"width={WIDTH},height={HEIGHT}",
        "!",
        "filesink",
        f"location={OUTPUT}",
    ]


def main() -> int:
    print(f"Recording from {DEVICE} -> {OUTPUT}")
    print("Recording raw YUY2 (lossless). Press Ctrl+C to stop.")
    print("Warning: file size is very large (~4 MB/frame at 1920x1080 YUY2).")

    status = run_pipeline(raw_pipeline())
    if status == 0:
        print(f"Saved: {OUTPUT}")
        print(
            "To play with ffplay: "
            f"ffplay -f rawvideo -pixel_format yuyv422 -video_size {WIDTH}x{HEIGHT} -framerate {FPS_NUM}/{FPS_DEN} {OUTPUT}"
        )
    else:
        print("ERROR: Recording failed.", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
