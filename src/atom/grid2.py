"""
This module defines the Grid2 class, which is used to represent a 2D grid.
"""

import numpy as np
import taichi as ti

from .math import roundup_power_of_2


class Grid:
    """
    This class represents a 2D grid. It stores the number of cells in each dimension, the origin of the grid, and the length of the sides of the cells.
    """

    def __init__(self):
        self.cell_2dcount = None
        self.origin = None
        self.cell_sides_length = None

    def to_numpy(self) -> np.ndarray:
        return np.array(
            [
                self.cell_2dcount[0],
                self.cell_2dcount[1],
                self.origin[0],
                self.origin[1],
                self.cell_sides_length,
            ],
            dtype=float,
        )

    def from_numpy(self, data: np.ndarray):
        self.cell_2dcount = np.array([data[0], data[1]], dtype=int)
        self.origin = np.array([data[2], data[3]])
        self.cell_sides_length = data[4]


@ti.func
def cell_center_point(cell_2dindex: ti.math.ivec2, origin, cell_sides_length: float):
    return origin + cell_sides_length * (cell_2dindex + 0.5)


@ti.func
def cell_2dindex_from_1dindex(cell_1dindex: int, cell_count_0: int) -> ti.math.ivec2:
    cell_2dindex = ti.math.ivec2(0)
    cell_2dindex[0] = cell_1dindex % cell_count_0
    cell_2dindex[1] = cell_1dindex // cell_count_0
    return cell_2dindex


@ti.func
def cell_2dindex_from_point(
    p: ti.math.vec2, origin, cell_sides_length: float
) -> ti.math.ivec2:
    cell_2dindex_float = (p - origin) / cell_sides_length
    cell_2dindex = ti.cast(ti.math.floor(cell_2dindex_float), ti.i32)
    return cell_2dindex


@ti.func
def is_valid_cell_2dindex(index, cell_2dcount):
    is_valid = 1
    if (index >= cell_2dcount).any() or (index < ti.math.vec2(0, 0)).any():
        is_valid = 0
    return is_valid


@ti.func
def cell_1dindex_from_2dindex(cell_2dindex: ti.math.ivec2, cell_count_0: int) -> int:
    return cell_2dindex[0] + cell_2dindex[1] * cell_count_0


@ti.func
def cell_diagonal_length(cell_sides_length):
    return cell_sides_length * ti.sqrt(2.0)
