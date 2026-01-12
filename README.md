# Real-time Face Detection (Haar cascades)

This small project runs a webcam face detector using OpenCV's Haar cascades.

Setup
-----

1. Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Run
---

From the `Face-ID` folder run:

```bash
python3 face_detection.py
```

Options
-------

- `--camera`: camera index (0 default)
- `--scale`: scale factor for detection (default 1.1)
- `--min-neighbors`: detection sensitivity (default 5)
- `--min-size`: minimum face size in pixels (default 30)

Notes
-----

- The script uses the built-in OpenCV haarcascade file via `cv2.data.haarcascades`.
- Allow camera access when prompted by your OS.
- To use a different cascade, pass `--cascade` with the filename present in OpenCV's data folder or give a full path.
