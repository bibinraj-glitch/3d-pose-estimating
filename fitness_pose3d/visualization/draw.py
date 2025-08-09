from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from exc

from pose.pose_estimator import PoseResult


class Drawer:
    def __init__(self) -> None:
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_styles = mp.solutions.drawing_styles
        self._mp_pose = mp.solutions.pose

    def annotate(self, frame_bgr: np.ndarray, result: Optional[PoseResult], hud: Dict) -> np.ndarray:
        out = frame_bgr.copy()
        if result is not None:
            # Landmarks are in normalized coordinates for 2D drawing
            landmarks = [
                mp.framework.formats.landmark_pb2.NormalizedLandmark(x=float(x), y=float(y), z=float(z), visibility=float(v))
                for (x, y, z), v in zip(result.normalized_xy_z, result.visibility)
            ]
            landmark_list = mp.framework.formats.landmark_pb2.NormalizedLandmarkList(landmark=landmarks)
            self._mp_drawing.draw_landmarks(
                image=out,
                landmark_list=landmark_list,
                connections=self._mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self._mp_styles.get_default_pose_landmarks_style(),
                connection_drawing_spec=self._mp_styles.get_default_pose_connections_style(),
            )

        self._draw_hud(out, hud)
        return out

    def _draw_hud(self, img: np.ndarray, hud: Dict) -> None:
        h, w = img.shape[:2]
        x, y = 10, 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.7
        color = (255, 255, 255)
        shadow = (0, 0, 0)

        lines = [
            f"Exercise: {hud.get('exercise','-')}",
            f"Stage: {hud.get('stage','-')}",
            f"Reps: {hud.get('reps',0)}",
            f"Angle: {int(hud.get('angle',0))}",
            f"FPS: {hud.get('fps',0.0):.1f}",
            f"{hud.get('feedback','')}"
        ]
        for i, text in enumerate(lines):
            yy = y + i * 24
            cv2.putText(img, text, (x + 1, yy + 1), font, scale, shadow, 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, yy), font, scale, color, 2, cv2.LINE_AA)