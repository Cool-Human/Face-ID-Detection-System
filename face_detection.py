#!/usr/bin/env python3
"""Real-time face detection using Haar cascades (OpenCV).

Usage:
  python3 face_detection.py
  python3 face_detection.py --camera 1 --scale 1.1 --min-neighbors 5
"""
import argparse
import time
import cv2


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face detection with Haar cascades")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--cascade", type=str, default="haarcascade_frontalface_default.xml",
                   help="Haar cascade filename (uses OpenCV's data folder by default)")
    p.add_argument("--scale", type=float, default=1.1, help="Scale factor for detectMultiScale")
    p.add_argument("--min-neighbors", type=int, default=5, help="minNeighbors for detectMultiScale")
    p.add_argument("--min-size", type=int, default=30, help="Minimum face size in pixels")
    p.add_argument("--width", type=int, default=640, help="Optional camera frame width")
    p.add_argument("--height", type=int, default=480, help="Optional camera frame height")
    return p.parse_args()


def main():
    args = parse_args()

    cascade_path = cv2.data.haarcascades + args.cascade
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise SystemExit(f"Failed to load cascade at {cascade_path}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = 0.0
    prev = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=args.scale,
                minNeighbors=args.min_neighbors,
                minSize=(args.min_size, args.min_size)
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # FPS calculation
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            cv2.putText(frame, f"Faces: {len(faces)}  FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Face Detection (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
