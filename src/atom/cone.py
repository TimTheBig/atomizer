from math import cos

import numpy as np
import taichi as ti

from . import math


class Cone:
    def __init__(self) -> None:
        self.origin: np.ndarray = None
        self.direction: np.ndarray = None
        self.half_angle = None
        self.cos_half_angle: float = None

    def update(self, origin: np.ndarray, direction: np.ndarray, opening_angle: float):
        self.origin = origin
        self.direction = direction
        self.half_angle = opening_angle * 0.5
        self.cos_half_angle = cos(self.half_angle)

    def is_point_inside(self, p: np.ndarray) -> bool:
        return cone2_is_point_inside_kernel(
            p, self.origin, self.direction, self.cos_half_angle
        )


@ti.func
def is_point_inside(p, o, d, cos_half_angle: float):
    """
    Assume d is normalized
    """
    # vector from origin to point
    op = p - o
    op_length = ti.math.length(op)
    is_inside = True
    if op_length > math.FLOAT_EPSILON:
        op_normalized = op / op_length
        cos_theta = ti.math.dot(op_normalized, d)

        if cos_theta < cos_half_angle or cos_theta <= 0.0:
            is_inside = False
    return is_inside


@ti.kernel
def cone2_is_point_inside_kernel(
    p: ti.math.vec2, o: ti.math.vec2, d: ti.math.vec2, cos_half_angle: float
) -> int:
    return is_point_inside(p, o, d, cos_half_angle)
