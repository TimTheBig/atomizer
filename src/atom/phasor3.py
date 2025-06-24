import numpy as np
import taichi as ti
from tqdm import tqdm

from . import grid3, math, random
from .direction import (
    cartesian_to_spherical,
    polar_to_cartesian,
    spherical_to_cartesian,
)

SPATIAL_WEIGHT_EXPONENT = 1
SPATIAL_FILTER_RADIUS = 2.0

ALIGNMENT_ITERATION_COUNT = 128
# Probability to ignore a neighbor
P_IGNORE_NEIGHBOR_START = 0.5
STOP_IGNORING_NEIGHBOR = ALIGNMENT_ITERATION_COUNT // 2
SEED = 1


class Field:
    def __init__(self) -> None:
        self.grid: grid3.Grid = None
        self.direction = None
        self.phase = None
        self.state = None

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        direction_np = self.direction.to_numpy()
        phase_np = self.phase.to_numpy()
        state_np = self.state.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["direction"] = direction_np
        dict_array["phase"] = phase_np
        dict_array["state"] = state_np

        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        shape = self.grid.cell_3dcount

        self.direction = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.phase = ti.field(dtype=ti.f32, shape=shape)
        self.state = ti.field(dtype=ti.u32, shape=shape)

        self.direction.from_numpy(dict_array["direction"])
        self.phase.from_numpy(dict_array["phase"])
        self.state.from_numpy(dict_array["state"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)


class MultigridAligner:
    def __init__(self) -> None:
        self.direction: list = None
        self.phase: list = None
        self.state: list = None
        self.multigrid: grid3.Multigrid = None
        self.iteration_number: int = None
        self.cos_period: float = None

    def allocate_from_field(self, f: Field, cos_period):
        self.multigrid = grid3.Multigrid()
        self.multigrid.create_from_grid(f.grid)
        self.cos_period = cos_period
        self.iteration_number = 0

        self.direction = []
        self.phase = []
        self.state = []
        for i in range(self.multigrid.level_count):
            shape_i = self.multigrid.cell_3dcount[i]
            if i != 0:
                direction_i = ti.Vector.field(
                    n=2, dtype=f.direction.dtype, shape=shape_i
                )
                phase_i0 = ti.field(dtype=f.phase.dtype, shape=shape_i)
                state_i = ti.field(dtype=f.state.dtype, shape=shape_i)
            else:
                direction_i = f.direction
                phase_i0 = f.phase
                state_i = f.state

            phase_i1 = ti.field(dtype=f.phase.dtype, shape=shape_i)

            self.direction.append(direction_i)
            self.phase.append([phase_i0, phase_i1])
            self.state.append(state_i)

    def align(self):
        level_count_m1 = self.multigrid.level_count - 1
        for level_i in tqdm(range(level_count_m1)):
            self.restrict(level_i)

        for shifter in tqdm(range(self.multigrid.level_count)):
            # level i from self.multigrid.level_count - 1 to 0
            level_i = level_count_m1 - shifter

            if level_i < level_count_m1:
                self.iteration_number = 0
                for _ in range(ALIGNMENT_ITERATION_COUNT):
                    self.align_one_level_one_time(level_i)

            if level_i > 0:
                self.prolong(level_i)

    def restrict(self, level):
        phasor3_field_restrict(
            self.direction[level],
            self.phase[level][0],
            self.state[level],
            self.multigrid.cell_sides_length[level],
            self.cos_period,
            self.direction[level + 1],
            self.phase[level + 1][0],
            self.state[level + 1],
        )

    def prolong(self, level):
        phasor3_field_prolong(
            self.direction[level],
            self.direction[level - 1],
            self.phase[level][0],
            self.state[level - 1],
            self.multigrid.cell_sides_length[level],
            self.cos_period,
            self.phase[level - 1][0],
        )

    def align_one_level_one_time(self, level):
        phasor3_field_align_one_level_one_time(
            self.direction[level],
            self.phase[level][0],
            self.state[level],
            self.multigrid.cell_sides_length[level],
            self.cos_period,
            self.iteration_number,
            self.phase[level][1],
        )
        self.phase[level][0], self.phase[level][1] = (
            self.phase[level][1],
            self.phase[level][0],
        )
        self.iteration_number += 1


@ti.kernel
def phasor3_field_restrict(
    direction: ti.template(),
    phase: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    cos_period: float,
    direction_restricted: ti.template(),
    phase_restricted: ti.template(),
    state_restricted: ti.template(),
):
    origin = ti.math.vec3(0.0)
    cell_sides_length_restricted = cell_sides_length * 2.0
    for i in ti.grouped(direction_restricted):
        j_block_origin = i * 2

        d_i = ti.math.vec3(0.0)
        constraint_count = 0
        masked_count = 0

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, direction.shape)
            if is_invalid_j:
                masked_count = masked_count + 1
                continue

            if ti.math.isnan(direction[j][0]):
                masked_count = masked_count + 1
                continue

            if is_phase_constrained(state[j]):
                constraint_count = constraint_count + 1

            d_j = spherical_to_cartesian(direction[j])
            d_i = d_i + d_j

        if masked_count == 8:
            direction_restricted[i] = ti.math.vec2(ti.math.nan)
            continue

        d_i = math.normalize_safe(d_i)
        direction_restricted[i] = cartesian_to_spherical(d_i)

        if constraint_count == 0:
            continue

        state_restricted[i] = constrain_phase(state_restricted[i])

        p_i = grid3.cell_center_point(i, origin, cell_sides_length_restricted)
        phasor3_i = math.vec6(p_i, direction_restricted[i], cos_period)

        phase_c_average = ti.math.vec2(0.0)

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, direction.shape)
            if is_invalid_j:
                continue

            if ti.math.isnan(direction[j][0]):
                continue

            if not is_phase_constrained(state[j]):
                continue

            p_j = grid3.cell_center_point(j, origin, cell_sides_length)
            phasor3_j = math.vec7(p_j, direction[j], phase[j], cos_period)
            phase_ij = align_phase_i_with_j(phasor3_i, phasor3_j)
            d_j = spherical_to_cartesian(direction[j])
            alignment_ij = ti.abs(ti.math.dot(d_i, d_j))
            phase_c_average = phase_c_average + alignment_ij * polar_to_cartesian(
                phase_ij
            )

        phase_average = ti.math.atan2(phase_c_average.y, phase_c_average.x)
        phase_restricted[i] = phase_average


@ti.kernel
def phasor3_field_prolong(
    direction: ti.template(),
    direction_prolonged: ti.template(),
    phase: ti.template(),
    state_prolonged: ti.template(),
    cell_sides_length: float,
    cos_period: float,
    phase_prolonged: ti.template(),
):
    """
    in: direction, direction_prolonged, phase, state_prolonged
    in/out: phase_prolonged
    """
    origin = ti.math.vec3(0.0)
    cell_sides_length_prolonged = cell_sides_length * 0.5
    for i in ti.grouped(direction):
        is_masked_i = ti.math.isnan(direction[i][0])
        if is_masked_i:
            continue

        j_block_origin = i * 2

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        phasor_i = math.vec7(p_i, direction[i], phase[i], cos_period)

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, direction_prolonged.shape)
            if is_invalid_j:
                continue

            is_masked_j = ti.math.isnan(direction_prolonged[j][0])
            if is_masked_j:
                continue

            if not is_phase_constrained(state_prolonged[j]):
                p_j = grid3.cell_center_point(j, origin, cell_sides_length_prolonged)
                phasor_j = math.vec6(p_j, direction_prolonged[j], cos_period)
                phase_j = align_phase_i_with_j(phasor_j, phasor_i)
                phase_prolonged[j] = phase_j


@ti.kernel
def phasor3_field_align_one_level_one_time(
    direction: ti.template(),
    phase_in: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    cos_period: float,
    iteration_number: int,
    phase_out: ti.template(),
):
    origin = ti.math.vec3(0.0)

    for i in ti.grouped(direction):
        # If masked
        if ti.math.isnan(direction[i][0]):
            phase_out[i] = phase_in[i]
            continue

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        d_i = spherical_to_cartesian(direction[i])
        phasor_i = math.vec6(p_i, direction[i], cos_period)

        phase_c_average = ti.math.vec2(0.0)

        for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
            j = i + shifter - ti.math.ivec3(1, 1, 1)

            if not grid3.is_valid_cell_3dindex(j, direction.shape) or (i == j).all():
                continue

            if ti.math.isnan(direction[j][0]):
                continue

            p_ignore_neighbor = P_IGNORE_NEIGHBOR_START * (
                1.0 - ti.math.min(iteration_number / STOP_IGNORING_NEIGHBOR, 1.0)
            )

            if p_ignore_neighbor > 0.0:
                random_float = random.pcgf_7_to_1(
                    math.uvec7(ti.u32(SEED + iteration_number), i, j)
                )
                if random_float < p_ignore_neighbor:
                    continue

            p_j = grid3.cell_center_point(j, origin, cell_sides_length)
            w_ij = math.eval_triangle_filter_normalized(
                p_j,
                p_i,
                SPATIAL_FILTER_RADIUS * grid3.cell_diagonal_length(cell_sides_length),
            )
            w_ij = w_ij**SPATIAL_WEIGHT_EXPONENT

            phasor_j = math.vec7(p_j, direction[j], phase_in[j], cos_period)
            phase_ij = align_phase_i_with_j(phasor_i, phasor_j)
            d_j = spherical_to_cartesian(direction[j])
            alignment_ij = ti.abs(ti.math.dot(d_i, d_j))
            phase_c_average = (
                phase_c_average + w_ij * alignment_ij * polar_to_cartesian(phase_ij)
            )

        phase_average = ti.math.atan2(phase_c_average.y, phase_c_average.x)
        phase_out[i] = phase_average

        if is_phase_constrained(state[i]):
            phase_out[i] = phase_in[i]


@ti.func
def align_phase_i_with_j(i: math.vec6, j: math.vec7) -> float:
    p_i = i[:3]
    spherical_direction_i = i[3:5]
    period_i = i[5]

    p_j = j[:3]
    spherical_direction_j = j[3:5]
    phase_j = j[5]
    period_j = j[6]

    f_i = 1.0 / period_i
    f_j = 1.0 / period_j
    d_i = spherical_to_cartesian(spherical_direction_i)
    d_j = spherical_to_cartesian(spherical_direction_j)
    is_inverse_direction = ti.math.dot(d_i, d_j) < 0.0

    p_ij = (p_i + p_j) * 0.5
    a = 2.0 * ti.math.pi * f_i * ti.math.dot(d_i, p_ij - p_i)
    b = 2.0 * ti.math.pi * f_j * ti.math.dot(d_j, p_ij - p_j) + phase_j
    # Solve a + phase_i = b for phase_i
    # i.e., modify the phase to have the same angle at p_ij
    # inverse the angle of b if the directions are inversed
    # By doing so, we inverse the direction of increase of the function as
    # wanted and we still keep the same roots
    phase_i = b - a
    phase_i_inv = -b - a
    if is_inverse_direction:
        phase_i = phase_i_inv
    return phase_i


@ti.func
def eval_angle(
    x: ti.math.vec3,
    point: ti.math.vec3,
    direction: ti.math.vec3,
    phase: float,
    period: float,
) -> float:
    angle = ti.math.pi * 2.0 / period * ti.math.dot(direction, x - point) + phase
    return angle


@ti.func
def field_eval(
    direction: ti.template(),
    phase: ti.template(),
    x: ti.math.vec3,
    cell_sides_length: float,
    cos_period: float,
) -> float:
    """
    Result
    ------
    float

    Between 0 and 1
    """
    origin = ti.math.vec3(0.0)
    i = grid3.cell_3dindex_from_point(x, origin, cell_sides_length)

    sum_weight = 0.0
    sum_weighted_cos = 0.0

    for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
        j = i + shifter - ti.math.ivec3(1, 1, 1)
        j_is_valid = grid3.is_valid_cell_3dindex(j, direction.shape)
        if not j_is_valid:
            continue

        if ti.math.isnan(direction[j][0]):
            continue

        p_j = grid3.cell_center_point(j, origin, cell_sides_length)
        d_j = spherical_to_cartesian(direction[j])
        angle_j = eval_angle(x, p_j, d_j, phase[j], cos_period)
        cos_angle_j = ti.cos(angle_j)

        w_ij = math.eval_triangle_filter_normalized(
            x,
            p_j,
            cell_sides_length,
        )
        w_ij = w_ij**SPATIAL_WEIGHT_EXPONENT

        sum_weight += w_ij
        sum_weighted_cos += w_ij * cos_angle_j

    cos_val = sum_weighted_cos / sum_weight

    i_is_valid = grid3.is_valid_cell_3dindex(i, direction.shape)
    if not i_is_valid:
        cos_val = ti.math.nan
    else:
        if ti.math.isnan(direction[i][0]):
            cos_val = ti.math.nan

    return cos_val


@ti.func
def constrain_phase(state: ti.u32) -> ti.u32:
    state |= 0b001
    return state


@ti.func
def unconstrain_phase(state: ti.u32) -> ti.u32:
    state &= 0b110
    return state


@ti.func
def is_phase_constrained(state: ti.u32) -> int:
    mask = 0b001
    masked = mask & state
    return masked == mask
