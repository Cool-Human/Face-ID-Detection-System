#!/usr/bin/env python3
"""Real-time face detection using YuNET (OpenCV DNN).

YuNET is a lightweight face detector that's faster and more accurate than Haar cascades.
Requires: opencv-python >= 4.8.0

Usage:
  python3 yunnet_face_detection_version_1.py
  python3 yunnet_face_detection_version_1.py --camera 1 --conf-threshold 0.6
"""
import argparse
import time
import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face detection with YuNET")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--conf-threshold", type=float, default=0.5,
                   help="Confidence threshold for detections (default: 0.5)")
    p.add_argument("--nms-threshold", type=float, default=0.4,
                   help="NMS threshold for duplicate detections (default: 0.4)")
    p.add_argument("--width", type=int, default=640, help="Optional camera frame width")
    p.add_argument("--height", type=int, default=480, help="Optional camera frame height")
    p.add_argument("--top-k", type=int, default=5000, help="Keep top K detections (default: 5000)")
    return p.parse_args()


def get_yunnet_model_path():
    """Get the path to YuNET model shipped with OpenCV."""
    import sys
    opencv_data_path = cv2.data.haarcascades.replace('haarcascades', 'dnn')
    model_path = opencv_data_path + 'face_detection_yunet_2023mar.onnx'
    return model_path


def main():
    args = parse_args()

    # Load YuNET model
    model_path = get_yunnet_model_path()
    print(f"Loading YuNET model from: {model_path}")
    face_detector = cv2.FaceDetectorYN.create(
        model_path,
        "",
        input_size=(320, 320),
        score_threshold=args.conf_threshold,
        nms_threshold=args.nms_threshold,
        top_k=args.top_k,
        keep_top_k=5000
    )

    if face_detector is None:
        raise SystemExit(
            f"Failed to load YuNET model. Ensure OpenCV >= 4.8.0 is installed and model path is correct.\n"
            f"Expected path: {model_path}"
        )

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    fps = 0.0
    prev = time.time()

    print("YuNET face detector initialized. Press 'q' or ESC to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame from camera")
                break

            # Set input size to match frame
            h, w = frame.shape[:2]
            face_detector.setInputSize((w, h))

            # Detect faces
            start_detect = time.time()
            _, faces = face_detector.detect(frame)
            detect_time = time.time() - start_detect

            # Draw detections
            if faces is not None and len(faces) > 0:
                for face in faces:
                    # face is [x, y, w, h, conf, l_eye_x, l_eye_y, r_eye_x, r_eye_y,
                    #         nose_x, nose_y, l_mouth_x, l_mouth_y, r_mouth_x, r_mouth_y]
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    conf = face[4]

                    # Draw bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Draw confidence score
                    cv2.putText(frame, f"{conf:.2f}", (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                    # Draw landmarks (eyes, nose, mouth)
                    landmarks = face[5:].reshape(-1, 2).astype(int)
                    for landmark in landmarks:
                        cv2.circle(frame, tuple(landmark), 2, (0, 0, 255), -1)

            # FPS calculation
            now = time.time()
            dt = now - prev
            prev = now
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)

            # Overlay statistics
            num_faces = len(faces) if faces is not None else 0
            cv2.putText(frame, f"Faces: {num_faces}  FPS: {fps:.1f}  Detect: {detect_time*1000:.1f}ms",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("YuNET Face Detection (press q to quit)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
