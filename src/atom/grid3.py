import numpy as np
import taichi as ti

from . import math


class Grid:
    def __init__(self) -> None:
        self.cell_3dcount = None
        self.origin = None
        self.cell_sides_length = None

    def to_numpy(self) -> np.ndarray:
        return np.concatenate(
            (
                self.cell_3dcount.astype(np.float32),
                self.origin.astype(np.float32),
                np.array([self.cell_sides_length], dtype=np.float32),
            )
        )

    def from_numpy(self, data: np.ndarray):
        self.cell_3dcount = np.array(data[0:3], dtype=int)
        self.origin = np.array(data[3:6])
        self.cell_sides_length = data[6]


class Multigrid:
    def __init__(self) -> None:
        self.cell_3dcount = None
        self.origin = None
        self.cell_sides_length = None
        self.level_count = None

    def create_from_grid(self, grid: Grid):
        self.origin = grid.origin

        self.cell_3dcount = []
        self.level_count = 0
        self.cell_3dcount.append(grid.cell_3dcount)
        for axis_index in range(3):
            level_number = 0
            roundup_power_of_2_level_i = int(
                math.roundup_power_of_2(int(grid.cell_3dcount[axis_index]))
            )
            while roundup_power_of_2_level_i > 0:
                if level_number > 0:
                    # Check if a cell_3dcount is allocated for this level
                    if len(self.cell_3dcount) < level_number + 1:
                        self.cell_3dcount.append(np.array([1, 1, 1]))

                    self.cell_3dcount[level_number][
                        axis_index
                    ] = roundup_power_of_2_level_i

                level_number += 1
                roundup_power_of_2_level_i = roundup_power_of_2_level_i // 2

            self.level_count = len(self.cell_3dcount)

        self.cell_sides_length = [
            grid.cell_sides_length * 2 ** (level) for level in range(self.level_count)
        ]


@ti.func
def cell_center_point(cell_3dindex: ti.math.ivec3, origin, cell_sides_length: float):
    return origin + cell_sides_length * (cell_3dindex + 0.5)


@ti.func
def cell_3dindex_from_1dindex(
    cell_1dindex: int, cell_3dcount: ti.math.ivec3
) -> ti.math.ivec3:
    cell_3dindex = ti.math.ivec3(0)
    cell_3dindex[0] = cell_1dindex % cell_3dcount[0]
    cell_3dindex[1] = (cell_1dindex // cell_3dcount[0]) % cell_3dcount[1]
    cell_3dindex[2] = (
        cell_1dindex // (cell_3dcount[0] * cell_3dcount[1])
    ) % cell_3dcount[2]
    return cell_3dindex


@ti.func
def cell_3dindex_from_point(
    p: ti.math.vec3, origin, cell_sides_length: float
) -> ti.math.ivec3:
    cell_3dindex_float = (p - origin) / cell_sides_length
    cell_3dindex = ti.cast(ti.math.floor(cell_3dindex_float), ti.i32)
    return cell_3dindex


@ti.func
def cell_1dindex_from_3dindex(
    cell_3dindex: ti.math.ivec3, cell_count_xy: ti.math.ivec2
) -> int:
    return (
        cell_3dindex[0]
        + cell_3dindex[1] * cell_count_xy[0]
        + cell_3dindex[2] * cell_count_xy[0] * cell_count_xy[1]
    )


@ti.func
def is_valid_cell_3dindex(index, cell_3dcount):
    is_valid = 1
    if (index >= cell_3dcount).any() or (index < ti.math.ivec3(0)).any():
        is_valid = 0
    return is_valid


@ti.func
def cell_diagonal_length(cell_sides_length):
    return cell_sides_length * ti.sqrt(3.0)
