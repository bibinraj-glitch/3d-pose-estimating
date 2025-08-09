import argparse
import time
from collections import deque

import cv2
import numpy as np

from pose.pose_estimator import PoseEstimator
from metrics.exercise_counters import create_exercise_counter
from visualization.draw import Drawer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam 3D pose fitness demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default 0)")
    parser.add_argument("--exercise", type=str, default="squat", choices=["squat", "pushup", "curl"], help="Exercise to count")
    parser.add_argument("--model-complexity", type=int, default=1, choices=[0, 1, 2], help="MediaPipe model complexity")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--no-draw", action="store_true", help="Disable drawing for speed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    pose_estimator = PoseEstimator(
        model_complexity=args.model_complexity,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    counter = create_exercise_counter(args.exercise)
    drawer = Drawer()

    fps_hist = deque(maxlen=30)
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            t0 = time.time()
            result = pose_estimator.process_bgr(frame_bgr)

            hud = {
                "exercise": args.exercise,
                "reps": counter.state.rep_count,
                "stage": counter.state.stage,
                "fps": 0.0,
                "angle": counter.state.last_angle,
                "feedback": getattr(counter, 'feedback', ''),
            }

            if result is not None:
                counter.update(result)
                hud["reps"] = counter.state.rep_count
                hud["stage"] = counter.state.stage

            t1 = time.time()
            fps = 1.0 / max(1e-6, (t1 - t0))
            fps_hist.append(fps)
            hud["fps"] = float(np.mean(fps_hist)) if fps_hist else fps

            if not args.no_draw:
                annotated = drawer.annotate(frame_bgr, result, hud)
            else:
                annotated = frame_bgr

            cv2.imshow("Fitness Pose 3D (Webcam)", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose_estimator.close()


if __name__ == "__main__":
    main()