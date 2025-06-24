import numpy as np
import taichi as ti
from tqdm import tqdm

from . import grid3, math, random

SPATIAL_WEIGHT_EXPONENT = 1
SPATIAL_FILTER_RADIUS = 2.0

ALIGNMENT_ITERATION_COUNT = 64
# Probability to ignore a neighbor
P_IGNORE_NEIGHBOR_START = 0.5
STOP_IGNORING_NEIGHBOR = ALIGNMENT_ITERATION_COUNT // 2
SEED = 1


class SphericalField:
    def __init__(self):
        self.grid: grid3.Grid = None
        self.direction = None
        self.state = None

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        direction_np = self.direction.to_numpy()
        state_np = self.state.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["direction"] = direction_np
        dict_array["state"] = state_np
        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        self.direction = ti.Vector.field(
            n=2, dtype=ti.f32, shape=dict_array["state"].shape
        )
        self.state = ti.field(dtype=ti.u32, shape=dict_array["state"].shape)

        self.direction.from_numpy(dict_array["direction"])
        self.state.from_numpy(dict_array["state"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)


class SphericalMultigridAligner:
    def __init__(self):
        self.direction: list = None
        self.state: list = None
        self.multigrid: grid3.Multigrid = None
        self.iteration_number: int = None

    def allocate_from_field(self, f: SphericalField):
        self.multigrid = grid3.Multigrid()
        self.multigrid.create_from_grid(f.grid)

        self.direction = []
        self.state = []
        for i in range(self.multigrid.level_count):
            shape_i = self.multigrid.cell_3dcount[i]
            if i != 0:
                direction_i0 = ti.Vector.field(
                    n=2, dtype=f.direction.dtype, shape=shape_i
                )
                state_i = ti.field(dtype=f.state.dtype, shape=shape_i)
            else:
                direction_i0 = f.direction
                state_i = f.state

            direction_i1 = ti.Vector.field(n=2, dtype=f.direction.dtype, shape=shape_i)

            self.direction.append([direction_i0, direction_i1])
            self.state.append(state_i)

        self.iteration_number = 0

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
        """
        Assume a valid level and level + 1
        """
        spherical_field_restrict(
            self.direction[level][0],
            self.state[level],
            self.direction[level + 1][0],
            self.state[level + 1],
        )

    def prolong(self, level):
        """
        Assume a valid level and level - 1
        """
        spherical_field_prolong(
            self.direction[level][0],
            self.state[level - 1],
            self.direction[level - 1][0],
        )

    def align_one_level_one_time(self, level, smooth_constraints: bool = False):
        spherical_field_align_one_level_one_time(
            self.direction[level][0],
            self.state[level],
            self.multigrid.cell_sides_length[level],
            self.iteration_number,
            self.direction[level][1],
            smooth_constraints,
        )
        self.direction[level][0], self.direction[level][1] = (
            self.direction[level][1],
            self.direction[level][0],
        )
        self.iteration_number += 1


@ti.kernel
def spherical_field_restrict(
    direction: ti.template(),
    state: ti.template(),
    direction_restricted: ti.template(),
    state_restricted: ti.template(),
):
    """
    in: orientation, state
    out: direction_restricted, state_restricted
    """
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

            if not is_constrained(state[j]):
                continue

            constraint_count = constraint_count + 1
            d_j = spherical_to_cartesian(direction[j])
            d_i = d_i + d_j

        if constraint_count > 0:
            d_i = math.normalize_safe(d_i)
            direction_restricted[i] = cartesian_to_spherical(d_i)
            state_restricted[i] = constrain(state_restricted[i])
        if masked_count == 8:
            direction_restricted[i] = ti.math.vec2(ti.math.nan)


@ti.kernel
def spherical_field_prolong(
    direction: ti.template(),
    state_prolonged: ti.template(),
    direction_prolonged: ti.template(),
):
    """
    in: direction, state_prolonged
    in/out: direction_prolonged
    """
    for i in ti.grouped(direction):
        is_masked_i = ti.math.isnan(direction[i][0])
        if is_masked_i:
            continue

        j_block_origin = i * 2

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, direction_prolonged.shape)
            if is_invalid_j:
                continue

            is_masked_j = ti.math.isnan(direction_prolonged[j][0])
            if is_masked_j:
                continue

            if not is_constrained(state_prolonged[j]):
                direction_prolonged[j] = direction[i]


@ti.kernel
def spherical_field_align_one_level_one_time(
    direction_in: ti.template(),
    state_in: ti.template(),
    cell_sides_length: float,
    iteration_number: int,
    direction_out: ti.template(),
    smooth_constraints: int,
):
    origin = ti.math.vec3(0.0)

    for i in ti.grouped(direction_in):
        # If masked
        if ti.math.isnan(direction_in[i][0]):
            direction_out[i] = direction_in[i]
            continue

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        d_average = ti.math.vec3(0.0)

        for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
            j = i + shifter - ti.math.ivec3(1, 1, 1)

            if not grid3.is_valid_cell_3dindex(j, direction_in.shape) or (i == j).all():
                continue

            if ti.math.isnan(direction_in[j][0]):
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

            d_j = spherical_to_cartesian(direction_in[j])
            # DEBUG
            # print(f"w_ij: {w_ij}; d_j: {d_j}")
            d_average = d_average + w_ij * d_j

        d_average = math.normalize_safe(d_average)

        sph_d_average = cartesian_to_spherical(d_average)
        if is_constrained(state_in[i]) and not smooth_constraints:
            sph_d_average = direction_in[i]
        direction_out[i] = sph_d_average


@ti.func
def sample_uniform_sphere(u: ti.math.uvec2) -> ti.math.vec3:
    z = 1.0 - 2.0 * u[0]
    one_m_sqr_z = ti.max(0.0, 1.0 - z**2)
    r = ti.sqrt(one_m_sqr_z)
    phi = 2.0 * ti.math.pi * u[1]
    return ti.math.vec3(r * ti.cos(phi), r * ti.sin(phi), z)


@ti.func
def polar_to_cartesian(angle):
    return ti.math.vec2(ti.cos(angle), ti.sin(angle))


@ti.func
def cartesian_to_polar(v):
    return ti.atan2(v.y, v.x)


@ti.func
def sample_uniform_circle(u: ti.u32) -> float:
    random.pcgf(u) * 2.0 * ti.math.pi


@ti.func
def diff_normalized(d0, d1) -> float:
    return (-1 * ti.math.dot(d0, d1) + 1) * 0.5


@ti.func
def spherical_to_cartesian(d: ti.math.vec2) -> ti.math.vec3:
    theta = d[0]
    phi = d[1]
    return ti.math.vec3(
        ti.math.cos(phi) * ti.math.sin(theta),
        ti.math.sin(phi) * ti.math.sin(theta),
        ti.math.cos(theta),
    )


@ti.kernel
def spherical_to_cartesian_kernel(d: ti.math.vec2) -> ti.math.vec3:
    return spherical_to_cartesian(d)


@ti.func
def orthogonolize(x, n):
    """
    Orthogonolize the direction x with n by projecting x on the plane defined
    by normal n.

    Notes
    -----
    https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process
    """
    return math.normalize_safe(x - (ti.math.dot(n, x) / ti.math.dot(n, n)) * n)


@ti.func
def nlerp(x, y, t):
    return math.normalize_safe(ti.math.mix(x, y, t))


@ti.kernel
def nlerp_kernel(x: ti.math.vec3, y: ti.math.vec3, t: float) -> ti.math.vec3:
    return nlerp(x, y, t)


@ti.func
def slerp(p0: ti.math.vec3, p1: ti.math.vec3, t: float):
    # Clamp dot product to avoid numerical issues with acos
    dot = ti.math.dot(p0, p1)
    omega = math.acos_safe(dot)

    result = ti.math.vec3(0.0)
    if omega < 1e-5:
        # If angle is very small, fallback to linear interpolation (lerp)
        result = (1.0 - t) * p0 + t * p1
    else:
        sin_omega = ti.sin(omega)
        s0 = ti.sin((1.0 - t) * omega) / sin_omega
        s1 = ti.sin(t * omega) / sin_omega
        result = s0 * p0 + s1 * p1

    return result


@ti.func
def cartesian_to_spherical(d: ti.math.vec3) -> ti.math.vec2:
    return ti.math.vec2(math.acos_safe(d.z), ti.math.atan2(d.y, d.x))


@ti.kernel
def cartesian_to_spherical_kernel(d: ti.math.vec3) -> ti.math.vec2:
    return cartesian_to_spherical(d)


@ti.func
def constrain(state: ti.u32) -> ti.u32:
    state |= 0b01
    return state


@ti.func
def unconstrain(state: ti.u32) -> ti.u32:
    state &= 0b10
    return state


@ti.func
def is_constrained(state: ti.u32) -> int:
    mask = 0b01
    masked = mask & state
    return masked == mask
