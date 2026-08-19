"""Record a short clip from the webcam for calibration."""
import argparse
import time
from pathlib import Path

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="mm/calib.mp4")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=6.0)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if fps <= 1 or fps > 120:
        fps = 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera {args.camera}: {w}x{h} @ {fps:.0f} fps")
    print("Position BOTH reference objects flat on the floor, in view.")
    print("SPACE = start recording   |   q = quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            cap.release(); raise SystemExit("Camera read failed")
        preview = frame.copy()
        cv2.putText(preview, "SPACE to record, q to quit", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("calibration setup", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            break
        if key == ord("q"):
            cap.release(); cv2.destroyAllWindows(); raise SystemExit("Cancelled")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release(); cv2.destroyAllWindows()
        raise SystemExit("Could not open video writer")

    print(f"Recording {args.seconds:.0f}s -> {out_path} ... hold still")
    t0, n = time.time(), 0
    while time.time() - t0 < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame); n += 1
        cv2.imshow("calibration setup", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    writer.release(); cap.release(); cv2.destroyAllWindows()
    print(f"Saved {n} frames -> {out_path}")


if __name__ == "__main__":
    main()
