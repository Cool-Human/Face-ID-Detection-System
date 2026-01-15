#!/usr/bin/env python3
"""Real-time face detection using YuNET (OpenCV DNN).

Usage:
  python3 face_detection_version_1.py
  python3 face_detection_version_1.py --camera 1 --scale 0.5 --conf-threshold 0.9
"""
import argparse
import os
import time
import urllib.request
import cv2
import numpy as np


def download_model(model_path, url):
    """Download the model if it doesn't exist."""
    if not os.path.exists(model_path):
        print(f"Downloading model from {url}...")
        urllib.request.urlretrieve(url, model_path)
        print("Model downloaded.")


def parse_args():
    p = argparse.ArgumentParser(description="Real-time face detection with YuNET")
    p.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--model", type=str, default="face_detection_yunet_2023mar.onnx",
                   help="YuNET model filename")
    p.add_argument("--scale", type=float, default=1.0, help="Scale factor for input image")
    p.add_argument("--conf-threshold", type=float, default=0.9, help="Confidence threshold for detection")
    p.add_argument("--nms-threshold", type=float, default=0.3, help="NMS threshold")
    p.add_argument("--top-k", type=int, default=5000, help="Keep top k detections")
    p.add_argument("--width", type=int, default=160, help="Camera frame width")
    p.add_argument("--height", type=int, default=120, help="Camera frame height")
    return p.parse_args()


def main():
    args = parse_args()

    # Model URL
    model_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
    model_path = args.model

    # Download model if not present
    download_model(model_path, model_url)

    # Load the model
    detector = cv2.FaceDetectorYN.create(model_path, "", (args.width, args.height), args.conf_threshold, args.nms_threshold, args.top_k)
    if detector is None:
        raise SystemExit("Failed to create FaceDetectorYN")

    # Open camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open camera {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    prev_time = time.time()

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame")
            break

        current_time = time.time()
        fps = 1 / (current_time - prev_time) if current_time - prev_time > 0 else 0
        prev_time = current_time

        # Resize frame to model input size
        input_frame = cv2.resize(frame, (args.width, args.height))

        # Detect faces
        faces = detector.detect(input_frame)

        # Scale factors
        scale_x = frame.shape[1] / args.width
        scale_y = frame.shape[0] / args.height

        num_faces = len(faces[1]) if faces[1] is not None else 0

        if faces[1] is not None:
            for face in faces[1]:
                # Draw bounding box
                bbox = face[:4].astype(int)
                bbox[0] = int(bbox[0] * scale_x)
                bbox[1] = int(bbox[1] * scale_y)
                bbox[2] = int(bbox[2] * scale_x)
                bbox[3] = int(bbox[3] * scale_y)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), (0, 255, 0), 2)

                # Draw landmarks
                landmarks = face[4:14].reshape(5, 2).astype(int)
                for lm in landmarks:
                    lm[0] = int(lm[0] * scale_x)
                    lm[1] = int(lm[1] * scale_y)
                    cv2.circle(frame, tuple(lm), 2, (255, 0, 0), -1)

        # Display counter
        text = f"Faces: {num_faces} FPS: {fps:.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Face Detection with YuNET", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()