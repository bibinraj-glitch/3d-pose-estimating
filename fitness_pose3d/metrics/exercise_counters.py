from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np

from utils.angles import EMAFilter, compute_angle_degrees
from utils.landmarks import LM, get as lm_get
from pose.pose_estimator import PoseResult


@dataclass
class CounterState:
    rep_count: int = 0
    stage: Literal["start", "down", "up"] = "start"
    last_angle: float = 180.0


class ExerciseCounter:
    def __init__(self, name: str, low_thresh: float, high_thresh: float, alpha: float = 0.2) -> None:
        self.name = name
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.ema = EMAFilter(alpha=alpha)
        self.state = CounterState()

    def compute_primary_angle(self, pose: PoseResult) -> Optional[float]:  # to override
        raise NotImplementedError

    def update(self, pose: Optional[PoseResult]) -> None:
        if pose is None:
            return
        angle = self.compute_primary_angle(pose)
        if angle is None:
            return
        angle_smooth = self.ema.update(angle)
        self.state.last_angle = angle_smooth

        # Simple hysteresis state machine
        if self.state.stage in ("start", "up"):
            if angle_smooth < self.low_thresh:
                self.state.stage = "down"
        elif self.state.stage == "down":
            if angle_smooth > self.high_thresh:
                self.state.stage = "up"
                self.state.rep_count += 1


class SquatCounter(ExerciseCounter):
    def __init__(self) -> None:
        # Knee angle: deep when < ~80, standing when > ~160
        super().__init__(name="squat", low_thresh=80.0, high_thresh=160.0, alpha=0.2)

    def compute_primary_angle(self, pose: PoseResult) -> Optional[float]:
        # Use the side with higher visibility around knee
        left_vis = pose.visibility[LM["left_knee"]]
        right_vis = pose.visibility[LM["right_knee"]]
        side = "left" if left_vis >= right_vis else "right"
        hip = lm_get(pose.world_xyz, f"{side}_hip")
        knee = lm_get(pose.world_xyz, f"{side}_knee")
        ankle = lm_get(pose.world_xyz, f"{side}_ankle")
        return compute_angle_degrees(hip, knee, ankle)


class PushupCounter(ExerciseCounter):
    def __init__(self) -> None:
        # Elbow angle: down when < ~70, up when > ~160
        super().__init__(name="pushup", low_thresh=70.0, high_thresh=160.0, alpha=0.25)

    def compute_primary_angle(self, pose: PoseResult) -> Optional[float]:
        left_vis = pose.visibility[LM["left_elbow"]]
        right_vis = pose.visibility[LM["right_elbow"]]
        side = "left" if left_vis >= right_vis else "right"
        shoulder = lm_get(pose.world_xyz, f"{side}_shoulder")
        elbow = lm_get(pose.world_xyz, f"{side}_elbow")
        wrist = lm_get(pose.world_xyz, f"{side}_wrist")
        return compute_angle_degrees(shoulder, elbow, wrist)


def create_exercise_counter(name: str) -> ExerciseCounter:
    name = name.lower()
    if name == "squat":
        return SquatCounter()
    if name == "pushup":
        return PushupCounter()
    raise ValueError(f"Unknown exercise {name}")