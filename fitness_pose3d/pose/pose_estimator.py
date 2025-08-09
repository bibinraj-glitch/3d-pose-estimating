from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as exc:  # pragma: no cover - import-time dependency
    raise RuntimeError(
        "mediapipe is required. Install with `pip install mediapipe`."
    ) from exc


@dataclass
class PoseResult:
    image_width: int
    image_height: int
    # 2D normalized landmark positions (x,y) in [0,1], z is relative depth in [~meters?]
    normalized_xy_z: np.ndarray  # shape (33, 3)
    # 3D world landmark positions in meters (x,y,z), origin near hips
    world_xyz: np.ndarray  # shape (33, 3)
    # per-landmark visibility (0-1)
    visibility: np.ndarray  # shape (33,)


class PoseEstimator:
    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            enable_segmentation=False,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def close(self) -> None:
        if self._pose:
            self._pose.close()

    def process_bgr(self, image_bgr: np.ndarray) -> Optional[PoseResult]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)
        if not results.pose_landmarks or not results.pose_world_landmarks:
            return None

        h, w = image_bgr.shape[:2]
        lm = results.pose_landmarks.landmark
        wlm = results.pose_world_landmarks.landmark

        normalized_xy_z = np.array([[p.x, p.y, p.z] for p in lm], dtype=np.float32)
        world_xyz = np.array([[p.x, p.y, p.z] for p in wlm], dtype=np.float32)
        visibility = np.array([p.visibility for p in lm], dtype=np.float32)

        return PoseResult(
            image_width=w,
            image_height=h,
            normalized_xy_z=normalized_xy_z,
            world_xyz=world_xyz,
            visibility=visibility,
        )

    @property
    def connections(self) -> List[tuple[int, int]]:
        return list(self._mp_pose.POSE_CONNECTIONS)