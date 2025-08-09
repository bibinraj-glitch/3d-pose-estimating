from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def compute_angle_degrees(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Returns the angle ABC in degrees using 3D coordinates.

    a, b, c are 3D points (numpy arrays of shape (3,)).
    """
    ba = a - b
    bc = c - b

    ba_norm = np.linalg.norm(ba) + 1e-8
    bc_norm = np.linalg.norm(bc) + 1e-8
    cos_theta = float(np.dot(ba, bc) / (ba_norm * bc_norm))
    cos_theta = max(-1.0, min(1.0, cos_theta))
    angle_rad = math.acos(cos_theta)
    return math.degrees(angle_rad)


@dataclass
class EMAFilter:
    alpha: float = 0.2
    value: float | None = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = float(x)
        else:
            self.value = self.alpha * float(x) + (1.0 - self.alpha) * self.value
        return self.value