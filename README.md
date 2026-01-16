# Real-time Face Detection

This project provides three face detection implementations using OpenCV:
- **Version 0**: Haar cascade classifiers (fast, lightweight, classical).
- **Version 1**: YuNET DNN detector (modern, accurate, robust).
- **Version 2**: Optimized YuNET DNN detector (with GPU support and multithreading).

## Common Setup

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Version 0: Haar Cascades (face_detection.py)

### Run

From the `Face-ID-Detection-System` folder:

```bash
python3 face_detection.py
```

### Options

- `--camera`: Camera device index (default: 0)
- `--scale`: Scale factor for detection (default: 1.1)
- `--min-neighbors`: Detection sensitivity; higher = fewer false positives (default: 5)
- `--min-size`: Minimum face size in pixels (default: 30)
- `--width`: Optional camera frame width (default: 640)
- `--height`: Optional camera frame height (default: 480)

### Notes

- Uses OpenCV's built-in Haar cascade via `cv2.data.haarcascades`.
- Fast and lightweight; good for resource-constrained environments.
- Sensitive to pose, occlusion, and extreme lighting conditions.
- Allow camera access when prompted by your OS.

### Example

```bash
python3 face_detection.py --scale 1.05 --min-neighbors 3 --min-size 50
```

---

## Version 1: YuNET DNN (face_detection_version_1.py)

### Requirements

YuNET requires **OpenCV >= 4.5.0**. The model is downloaded automatically on first run.

### Run

From the `Face-ID-Detection-System` folder:

```bash
python3 face_detection_version_1.py
```

### Options

- `--camera`: Camera device index (default: 0)
- `--model`: YuNET model filename (default: face_detection_yunet_2023mar.onnx)
- `--conf-threshold`: Confidence threshold for detections; higher = fewer false positives (default: 0.9)
- `--nms-threshold`: NMS threshold for duplicate detections (default: 0.3)
- `--top-k`: Keep top K detections (default: 5000)
- `--width`: Model input width (default: 160)
- `--height`: Model input height (default: 120)

### Notes

- Modern DNN-based detector shipped with OpenCV.
- Outputs facial landmarks (2 eyes, nose, 2 mouth corners) overlaid as blue dots.
- Displays face count and FPS in the top-left corner.
- Better accuracy and robustness than Haar cascades.
- Can leverage GPU acceleration if OpenCV is built with CUDA support.

### Example

```bash
python3 face_detection_version_1.py --conf-threshold 0.8 --nms-threshold 0.3
```

---

## Version 2: Optimized YuNET DNN (face_detection_version_2.py)

### Requirements

YuNET requires **OpenCV >= 4.5.0**. The model is downloaded automatically on first run. For GPU acceleration, OpenCV must be built with CUDA support.

### Run

From the `Face-ID-Detection-System` folder:

```bash
python3 face_detection_version_2.py
```

### Options

- `--camera`: Camera device index (default: 0)
- `--model`: YuNET model filename (default: face_detection_yunet_2023mar.onnx)
- `--conf-threshold`: Confidence threshold for detections; higher = fewer false positives (default: 0.9)
- `--nms-threshold`: NMS threshold for duplicate detections (default: 0.3)
- `--top-k`: Keep top K detections (default: 5000)
- `--width`: Model input width (default: 160)
- `--height`: Model input height (default: 120)
- `--use-gpu`: Use GPU acceleration if available

### Notes

- Optimized version of YuNET with asynchronous detection using multithreading for smoother real-time performance.
- Captures at higher resolution (640x480) for better input quality, then resizes for detection.
- Supports GPU acceleration via CUDA for significant speedups on compatible hardware.
- Outputs facial landmarks (2 eyes, nose, 2 mouth corners) overlaid as blue dots.
- Displays face count and FPS in the top-left corner.
- Better accuracy and robustness than Haar cascades, with improved efficiency over Version 1.

### Example

```bash
python3 face_detection_version_2.py --use-gpu --conf-threshold 0.8
```

---

## Comparison

| Feature | Haar Cascades (v0) | YuNET (v1) | Optimized YuNET (v2) |
|---------|-------------------|-----------|----------------------|
| Speed | Very fast | Fast | Very fast (with GPU/threading) |
| Accuracy | Moderate | High | High |
| Pose robustness | Low | High | High |
| Landmarks | No | Yes (5 points) | Yes (5 points) |
| Model size | None (built-in) | ~6 MB ONNX | ~6 MB ONNX |
| GPU support | No | Yes (with CUDA) | Yes (with CUDA) |
| Multithreading | No | No | Yes |
| OpenCV version | 3.0+ | 4.5.0+ | 4.5.0+ |
| Overlays | FPS | Faces + FPS | Faces + FPS |

---

## General Notes

- Allow camera access when prompted by your OS.
- Press `q` or `ESC` to exit any detector.
- For best results, use adequate lighting and frontal/near-frontal face poses.
- Version 0 shows real-time FPS overlaid on the video.
- Versions 1 and 2 show face count and FPS overlaid on the video.
