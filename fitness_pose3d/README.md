# Fitness Pose 3D – Mini Project

A minimal Python project for 3D human pose estimation using MediaPipe Pose, aimed at fitness form feedback and rep counting (squats and push-ups). Includes a webcam app and a video file processor with simple on-screen HUD.

## Setup

- Requirements: Python 3.9–3.11
- Install dependencies:

```bash
pip install -r requirements.txt
```

- Optional: If running on a headless server, prefer `app_video.py` (no display needed) or use OpenCV's headless builds.

## Usage

- Webcam (interactive preview):

```bash
python app_webcam.py --exercise squat
# or
python app_webcam.py --exercise pushup
```

- Video file (process and optionally save):

```bash
python app_video.py --input path/to/video.mp4 --exercise squat --output out.mp4
```

- Web (mobile-ready UI, runs on-device in browser):

```bash
uvicorn web.server:app --host 0.0.0.0 --port 7860
# open http://localhost:7860 on desktop, or phone via your LAN IP: http://<your-ip>:7860
```

- Common flags:
  - `--model-complexity {0,1,2}`: higher is more accurate but slower (default 1)
  - `--min-detection-confidence 0.5` and `--min-tracking-confidence 0.5`
  - `--no-draw` to disable landmark drawing (faster)

## What it does

- Runs MediaPipe Pose to get 2D and 3D landmarks (in meters via `world_landmarks`).
- Computes joint angles and smooths them with an EMA filter.
- Uses simple state machines to count reps for squats (knee angle) and push-ups (elbow angle).
- Overlays a HUD with FPS, stage (down/up), and rep count.

## Project structure

```
fitness_pose3d/
  app_webcam.py
  app_video.py
  requirements.txt
  README.md
  pose/
    pose_estimator.py
  metrics/
    exercise_counters.py
  utils/
    angles.py
    landmarks.py
  visualization/
    draw.py
```

## Notes

- MediaPipe `world_landmarks` are in an approximate metric 3D space where the origin is near the hips and scale depends on model calibration. They are still useful for relative angles and motion.
- Accuracy depends on camera placement and lighting. For best results, ensure the whole body is visible.
- This is a learning/demo project and not a medical or coaching tool.