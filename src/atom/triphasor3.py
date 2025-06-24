from math import prod

import numpy as np
import taichi as ti
from tqdm import tqdm

from . import basis3, direction, frame3, grid3, math, phasor3, random, transform3

# Parameters for tricosine field alignment
SPATIAL_WEIGHT_EXPONENT = 1
SPATIAL_FILTER_RADIUS = 2.0
ALIGNMENT_ITERATION_COUNT = 128
# Probability to ignore a neighbor
P_IGNORE_NEIGHBOR_START = 0.5
STOP_IGNORING_NEIGHBOR = ALIGNMENT_ITERATION_COUNT // 2
SEED = 1

# Parameters for extracting local maxima
JITTER = 0
SIDE_SAMPLE_COUNT = 4


class Field:
    def __init__(self) -> None:
        self.grid: grid3.Grid = None
        self.normal = None
        self.phi_t = None
        self.phase_t = None
        self.phase_b = None
        self.phase_n = None
        self.state = None

    def compute_memory_usage(self):
        # Return unit is bytes

        # 7 * 32 bits:
        # - Spherical normal: 2 * 32 bits
        # - Tangent angle: 1 * 32 bits
        # - Triphase: 3 * 32 bits
        # - State: 1 * 32 bits

        # Unit: bytes (that's why there is * 4)
        return prod(self.grid.cell_3dcount) * 7.0 * 4.0

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        normal_np = self.normal.to_numpy()
        phi_t_np = self.phi_t.to_numpy()
        phase_t_np = self.phase_t.to_numpy()
        phase_b_np = self.phase_b.to_numpy()
        phase_n_np = self.phase_n.to_numpy()
        state_np = self.state.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["normal"] = normal_np
        dict_array["phi_t"] = phi_t_np
        dict_array["phase_t"] = phase_t_np
        dict_array["phase_b"] = phase_b_np
        dict_array["phase_n"] = phase_n_np
        dict_array["state"] = state_np

        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        shape = self.grid.cell_3dcount

        self.normal = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.phi_t = ti.field(dtype=ti.f32, shape=shape)
        self.phase_t = ti.field(dtype=ti.f32, shape=shape)
        self.phase_b = ti.field(dtype=ti.f32, shape=shape)
        self.phase_n = ti.field(dtype=ti.f32, shape=shape)
        self.state = ti.field(dtype=ti.u32, shape=shape)

        self.normal.from_numpy(dict_array["normal"])
        self.phi_t.from_numpy(dict_array["phi_t"])
        self.phase_t.from_numpy(dict_array["phase_t"])
        self.phase_b.from_numpy(dict_array["phase_b"])
        self.phase_n.from_numpy(dict_array["phase_n"])
        self.state.from_numpy(dict_array["state"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def allocate_frame_field(self, frame_field: frame3.Field):
        frame_field.grid = self.grid
        frame_field.point = ti.Vector.field(
            n=3, dtype=ti.f32, shape=frame_field.grid.cell_3dcount
        )
        frame_field.normal = ti.Vector.field(
            n=2, dtype=ti.f32, shape=frame_field.grid.cell_3dcount
        )
        frame_field.phi_t = ti.field(dtype=ti.f32, shape=frame_field.grid.cell_3dcount)

        return ti.field(dtype=ti.f32, shape=frame_field.grid.cell_3dcount)

    def extract_frame_field(
        self,
        frame3_field: frame3.Field,
        triphasor_value,
        cos_triperiod,
        extraction_count: int = 2,
    ):
        frame3_field.normal = self.normal
        frame3_field.phi_t = self.phi_t
        if extraction_count == 2:
            field_get_cell_points_with_highest_val_2ex(
                self.normal,
                self.phi_t,
                self.phase_t,
                self.phase_b,
                self.phase_n,
                cos_triperiod,
                self.grid.cell_sides_length,
                frame3_field.point,
                triphasor_value,
            )
        else:
            field_get_cell_points_with_highest_val_1ex(
                self.normal,
                self.phi_t,
                self.phase_t,
                self.phase_b,
                self.phase_n,
                cos_triperiod,
                self.grid.cell_sides_length,
                frame3_field.point,
                triphasor_value,
            )

        field_keep_local_maxima(frame3_field.point, triphasor_value)


class MultigridAligner:
    def __init__(self) -> None:
        self.normal: list = None
        self.phi_t: list = None
        self.phase_t: list = None
        self.phase_b: list = None
        self.phase_n: list = None
        self.state: list = None
        self.multigrid: grid3.Multigrid = None
        self.iteration_number: int = None
        self.cos_triperiod: np.ndarray = None

    def compute_memory_usage(self):
        memory_usage = 0.0
        for i in range(self.multigrid.level_count):
            # Return unit: bytes

            shape_i = self.multigrid.cell_3dcount[i]

            # 9 * 32 bits:
            # - Spherical normal: 2 * 32 bits
            # - Tangent angle: 1 * 32 bits
            # - Triphase: 3 * 32 bits
            # - Buffer for tangent and bitangent phases: 2 * 32 bits
            # - State: 1 * 32 bits

            # Unit: bytes (that's why there is * 4)
            memory_usage += prod(shape_i) * 9.0 * 4.0

        return memory_usage

    def allocate_from_field(self, f: Field, cos_triperiod):
        self.multigrid = grid3.Multigrid()
        self.multigrid.create_from_grid(f.grid)
        self.cos_triperiod = cos_triperiod
        self.iteration_number = 0

        self.normal = []
        self.phi_t = []
        self.phase_t = []
        self.phase_b = []
        self.phase_n = []
        self.state = []

        for i in range(self.multigrid.level_count):
            shape_i = self.multigrid.cell_3dcount[i]
            if i != 0:
                normal_i = ti.Vector.field(n=2, dtype=ti.f32, shape=shape_i)
                phi_t_i = ti.field(dtype=ti.f32, shape=shape_i)
                phase_t_i0 = ti.field(dtype=ti.f32, shape=shape_i)
                phase_b_i0 = ti.field(dtype=ti.f32, shape=shape_i)
                phase_n_i = ti.field(dtype=ti.f32, shape=shape_i)
                state_i = ti.field(dtype=ti.u32, shape=shape_i)
            else:
                normal_i = f.normal
                phi_t_i = f.phi_t
                phase_t_i0 = f.phase_t
                phase_b_i0 = f.phase_b
                phase_n_i = f.phase_n
                state_i = f.state

            phase_t_i1 = ti.field(dtype=ti.f32, shape=shape_i)
            phase_b_i1 = ti.field(dtype=ti.f32, shape=shape_i)

            self.normal.append(normal_i)
            self.phi_t.append(phi_t_i)
            self.phase_t.append([phase_t_i0, phase_t_i1])
            self.phase_b.append([phase_b_i0, phase_b_i1])
            self.phase_n.append(phase_n_i)
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
        triphasor3_field_restrict(
            self.normal[level],
            self.phi_t[level],
            self.phase_t[level][0],
            self.phase_b[level][0],
            self.phase_n[level],
            self.state[level],
            self.multigrid.cell_sides_length[level],
            self.cos_triperiod,
            self.normal[level + 1],
            self.phi_t[level + 1],
            self.phase_t[level + 1][0],
            self.phase_b[level + 1][0],
            self.phase_n[level + 1],
            self.state[level + 1],
        )

    def prolong(self, level):
        triphasor3_field_prolong(
            self.normal[level],
            self.normal[level - 1],
            self.phi_t[level],
            self.phi_t[level - 1],
            self.phase_t[level][0],
            self.phase_b[level][0],
            self.phase_n[level],
            self.phase_n[level - 1],
            self.state[level - 1],
            self.multigrid.cell_sides_length[level],
            self.cos_triperiod,
            self.phase_t[level - 1][0],
            self.phase_b[level - 1][0],
        )

    def align_one_level_one_time(self, level):
        triphasor3_field_align(
            self.normal[level],
            self.phi_t[level],
            self.phase_t[level][0],
            self.phase_b[level][0],
            self.phase_n[level],
            self.state[level],
            self.cos_triperiod,
            self.multigrid.cell_sides_length[level],
            self.iteration_number,
            self.phase_t[level][1],
            self.phase_b[level][1],
        )
        self.phase_t[level][0], self.phase_t[level][1] = (
            self.phase_t[level][1],
            self.phase_t[level][0],
        )
        self.phase_b[level][0], self.phase_b[level][1] = (
            self.phase_b[level][1],
            self.phase_b[level][0],
        )
        self.iteration_number += 1


@ti.kernel
def triphasor3_field_restrict(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
    normal_restricted: ti.template(),
    phi_t_restricted: ti.template(),
    phase_t_restricted: ti.template(),
    phase_b_restricted: ti.template(),
    phase_n_restricted: ti.template(),
    state_restricted: ti.template(),
):
    origin = ti.math.vec3(0.0)
    cell_sides_length_restricted = cell_sides_length * 2.0
    for i in ti.grouped(normal_restricted):
        j_block_origin = i * 2

        n_i = ti.math.vec3(0.0)
        constraint_phase_b_count = 0
        masked_count = 0

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal.shape)
            if is_invalid_j:
                masked_count = masked_count + 1
                continue

            if ti.math.isnan(normal[j][0]):
                masked_count = masked_count + 1
                continue

            if is_constrained_phase_b(state[j]):
                constraint_phase_b_count = constraint_phase_b_count + 1

            n_j = direction.spherical_to_cartesian(normal[j])
            n_i = n_i + n_j

        if masked_count == 8:
            normal_restricted[i] = ti.math.vec2(ti.math.nan)
            continue

        n_i = math.normalize_safe(n_i)
        normal_restricted[i] = direction.cartesian_to_spherical(n_i)

        t_from_n_i = basis3.tangent_from_normal(n_i)
        b_from_n_i = ti.math.cross(n_i, t_from_n_i)
        world_to_tangent_i = transform3.compute_frame_to_canonical_matrix(
            t_from_n_i, b_from_n_i, n_i, ti.math.vec3(0.0, 0.0, 0.0)
        ).transpose()

        t_i_cov_00 = 0.0
        t_i_cov_01 = 0.0
        t_i_cov_11 = 0.0
        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal.shape)
            if is_invalid_j:
                continue

            if ti.math.isnan(normal[j][0]):
                continue

            # ts_j: j's tangent_space
            t_j_ts_j = ti.math.vec3(direction.polar_to_cartesian(phi_t[j]), 0.0)

            n_j = direction.spherical_to_cartesian(normal[j])
            t_from_n_j = basis3.tangent_from_normal(n_j)
            b_from_n_j = ti.math.cross(n_j, t_from_n_j)
            tangent_j_to_world = transform3.compute_frame_to_canonical_matrix(
                t_from_n_j, b_from_n_j, n_j, ti.math.vec3(0.0, 0.0, 0.0)
            )

            t_j_ts_i = transform3.apply_to_vector(
                world_to_tangent_i @ tangent_j_to_world, t_j_ts_j
            )

            t_i_cov_00 += t_j_ts_i[0] * t_j_ts_i[0]
            t_i_cov_01 += t_j_ts_i[0] * t_j_ts_i[1]
            t_i_cov_11 += t_j_ts_i[1] * t_j_ts_i[1]

        t_i_cov = ti.math.mat2(
            [
                [t_i_cov_00, t_i_cov_01],
                [t_i_cov_01, t_i_cov_11],
            ]
        )

        t_i = math.eigenvec2_with_highest_eigenvalue_iterative(t_i_cov)
        phi_t_i = ti.math.atan2(t_i.y, t_i.x)

        phi_t_restricted[i] = phi_t_i

        if constraint_phase_b_count == 0:
            continue

        state_restricted[i] = constrain_phase_b(state_restricted[i])

        p_i = grid3.cell_center_point(i, origin, cell_sides_length_restricted)

        triphasor3_i = math.vec9(
            p_i,
            normal_restricted[i],
            phi_t_restricted[i],
            phase_t_restricted[i],
            phase_b_restricted[i],
            phase_n_restricted[i],
        )

        phase_t_c_average = ti.math.vec2(0.0, 0.0)
        phase_b_c_average = ti.math.vec2(0.0, 0.0)
        phase_n_c_average = ti.math.vec2(0.0, 0.0)

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal.shape)
            if is_invalid_j:
                continue

            if ti.math.isnan(normal[j][0]):
                continue

            p_j = grid3.cell_center_point(j, origin, cell_sides_length)
            triphasor3_j = math.vec9(
                p_j,
                normal[j],
                phi_t[j],
                phase_t[j],
                phase_b[j],
                phase_n[j],
            )
            triphase_ij = align_triphase_i_with_j(
                triphasor3_i, triphasor3_j, cos_triperiod
            )
            phase_t_c_ij = direction.polar_to_cartesian(triphase_ij[0])
            phase_b_c_ij = direction.polar_to_cartesian(triphase_ij[1])
            phase_n_c_ij = direction.polar_to_cartesian(triphase_ij[2])

            phase_t_c_average = phase_t_c_average + phase_t_c_ij
            if is_constrained_phase_b(state[j]):
                phase_b_c_average = phase_b_c_average + phase_b_c_ij
            phase_n_c_average = phase_n_c_average + phase_n_c_ij

        phase_t_restricted[i] = ti.math.atan2(phase_t_c_average.y, phase_t_c_average.x)
        phase_b_restricted[i] = ti.math.atan2(phase_b_c_average.y, phase_b_c_average.x)
        phase_n_restricted[i] = ti.math.atan2(phase_n_c_average.y, phase_n_c_average.x)


@ti.kernel
def triphasor3_field_prolong(
    normal: ti.template(),
    normal_prolonged: ti.template(),
    phi_t: ti.template(),
    phi_t_prolonged: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    phase_n_prolonged: ti.template(),
    state_prolonged: ti.template(),
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
    phase_t_prolonged: ti.template(),
    phase_b_prolonged: ti.template(),
):
    origin = ti.math.vec3(0.0)
    cell_sides_length_prolonged = cell_sides_length * 0.5
    for i in ti.grouped(normal):
        is_masked_i = ti.math.isnan(normal[i][0])
        if is_masked_i:
            continue

        j_block_origin = i * 2

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        triphasor3_i = math.vec9(
            p_i, normal[i], phi_t[i], phase_t[i], phase_b[i], phase_n[i]
        )

        for shifter in ti.grouped(ti.ndrange(2, 2, 2)):
            j = j_block_origin + shifter

            # Check if j is valid
            is_invalid_j = not grid3.is_valid_cell_3dindex(j, normal_prolonged.shape)
            if is_invalid_j:
                continue

            is_masked_j = ti.math.isnan(normal_prolonged[j][0])
            if is_masked_j:
                continue

            p_j = grid3.cell_center_point(j, origin, cell_sides_length_prolonged)
            triphasor3_j = math.vec9(
                p_j,
                normal_prolonged[j],
                phi_t_prolonged[j],
                phase_t_prolonged[j],
                phase_b_prolonged[j],
                phase_n_prolonged[j],
            )
            triphase_j = align_triphase_i_with_j(
                triphasor3_j, triphasor3_i, cos_triperiod
            )

            phase_t_prolonged[j] = triphase_j[0]
            if not is_constrained_phase_b(state_prolonged[j]):
                phase_b_prolonged[j] = triphase_j[1]


@ti.kernel
def triphasor3_field_align(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t_in: ti.template(),
    phase_b_in: ti.template(),
    phase_n: ti.template(),
    state: ti.template(),
    cos_triperiod: ti.math.vec3,
    cell_sides_length: float,
    iteration_number: int,
    phase_t_out: ti.template(),
    phase_b_out: ti.template(),
):
    origin = ti.math.vec3(0.0)

    for i in ti.grouped(normal):
        phase_t_out[i] = phase_t_in[i]
        phase_b_out[i] = phase_b_in[i]
        # If masked
        if ti.math.isnan(normal[i][0]):
            continue

        p_i = grid3.cell_center_point(i, origin, cell_sides_length)
        triphasor3_i = math.vec9(
            p_i,
            normal[i],
            phi_t[i],
            phase_t_in[i],
            phase_b_in[i],
            phase_n[i],
        )

        phase_t_c_average = ti.math.vec2(0.0, 0.0)
        phase_b_c_average = ti.math.vec2(0.0, 0.0)

        for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
            j = i + shifter - ti.math.ivec3(1, 1, 1)

            if not grid3.is_valid_cell_3dindex(j, normal.shape) or (i == j).all():
                continue

            if ti.math.isnan(normal[j][0]):
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

            triphasor3_j = math.vec9(
                p_j,
                normal[j],
                phi_t[j],
                phase_t_in[j],
                phase_b_in[j],
                phase_n[j],
            )
            triphase_ij = align_triphase_i_with_j(
                triphasor3_i, triphasor3_j, cos_triperiod
            )
            phase_t_c_ij = direction.polar_to_cartesian(triphase_ij[0])
            phase_b_c_ij = direction.polar_to_cartesian(triphase_ij[1])

            phase_t_c_average = phase_t_c_average + phase_t_c_ij
            phase_b_c_average = phase_b_c_average + phase_b_c_ij

        phase_t_out[i] = ti.math.atan2(phase_t_c_average.y, phase_t_c_average.x)
        if not is_constrained_phase_b(state[i]):
            phase_b_out[i] = ti.math.atan2(phase_b_c_average.y, phase_b_c_average.x)


@ti.kernel
def field_get_cell_points_with_highest_val_2ex(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    cos_triperiod: ti.math.vec3,
    cell_sides_length: float,
    cell_point: ti.template(),
    cell_point_val: ti.template(),
):
    """
    Parameters
    ----------
    cell_point: vector field. n=3. shape: 3D.
        Will contain the points with the highest value. It is the first output.
    cell_point_val: scalar field. shape: 3D.
        The value of the cell point. It is the second output.

    Notes
    -----
    All the fields should have the same shape.
    """
    for i in ti.grouped(normal):
        if ti.math.isnan(normal[i][0]):
            cell_point[i].x = ti.math.nan
            continue
        # Point (XYZ) + Value (W)
        point_with_highest_val = ti.math.vec4(0.0)
        point_with_highest_val = field_cell_get_point_with_highest_val_2ex(
            normal,
            phi_t,
            phase_t,
            phase_b,
            phase_n,
            i,
            cell_sides_length,
            cos_triperiod,
        )
        cell_point[i] = point_with_highest_val.xyz
        cell_point_val[i] = point_with_highest_val.w


@ti.kernel
def field_get_cell_points_with_highest_val_1ex(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    cos_triperiod: ti.math.vec3,
    cell_sides_length: float,
    cell_point: ti.template(),
    cell_point_val: ti.template(),
):
    """
    Parameters
    ----------
    cell_point: vector field. n=3. shape: 3D.
        Will contain the points with the highest value. It is the first output.
    cell_point_val: scalar field. shape: 3D.
        The value of the cell point. It is the second output.

    Notes
    -----
    All the fields should have the same shape.
    """
    for i in ti.grouped(normal):
        if ti.math.isnan(normal[i][0]):
            cell_point[i].x = ti.math.nan
            continue
        # Point (XYZ) + Value (W)
        point_with_highest_val = ti.math.vec4(0.0)
        point_with_highest_val = field_cell_get_point_with_highest_val_1ex(
            normal,
            phi_t,
            phase_t,
            phase_b,
            phase_n,
            i,
            cell_sides_length,
            cos_triperiod,
        )
        cell_point[i] = point_with_highest_val.xyz
        cell_point_val[i] = point_with_highest_val.w


@ti.kernel
def field_keep_local_maxima(point: ti.template(), value: ti.template()):
    for i in ti.grouped(point):
        is_local_maximum = True

        for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
            j = i + shifter - ti.math.ivec3(1, 1, 1)

            if (j == i).all():
                continue

            neighbor_is_valid = grid3.is_valid_cell_3dindex(j, point.shape)
            if not neighbor_is_valid:
                continue

            p_i = point[j]
            if ti.math.isnan(p_i.x):
                continue

            if value[j] > value[i]:
                is_local_maximum = False

        if not is_local_maximum:
            # Write in component Y to avoid concurrent reading / writing the X
            # component, because the loop is in parallel.
            point[i].y = ti.math.nan


@ti.func
def field_cell_get_point_with_highest_val_2ex(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    cell_3dindex: ti.math.ivec3,
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
) -> ti.math.vec4:
    grid_origin = ti.math.vec3(0.0)
    cell_origin = grid_origin + cell_3dindex * cell_sides_length

    value_max = 0.0
    point_value_max = ti.math.vec3(0.0)

    for i in ti.grouped(
        ti.ndrange(SIDE_SAMPLE_COUNT, SIDE_SAMPLE_COUNT, SIDE_SAMPLE_COUNT)
    ):
        seed_i = ti.cast(
            ti.math.ivec3(
                cell_3dindex.x + SEED,
                cell_3dindex.y + 13 * SEED,
                cell_3dindex.z + 17 * SEED,
            ),
            ti.u32,
        )
        u3d = random.pcg3df(seed_i)
        if ti.static(not JITTER):
            u3d = ti.math.vec3(0.5)

        # Unit cell space
        eval_point = (i + u3d) / SIDE_SAMPLE_COUNT
        # World cell space
        eval_point = eval_point * cell_sides_length + cell_origin

        cos_triphase_i = field_eval(
            normal,
            phi_t,
            phase_t,
            phase_b,
            phase_n,
            eval_point,
            cell_sides_length,
            cos_triperiod,
        )
        cos_triphase_normalized_i = ti.abs(cos_triphase_i)
        triphasor3_val = (
            cos_triphase_normalized_i[0]
            + cos_triphase_normalized_i[1]
            + cos_triphase_normalized_i[2]
        ) / 3.0

        if triphasor3_val > value_max:
            value_max = triphasor3_val
            point_value_max = eval_point

    return ti.math.vec4(point_value_max, value_max)


@ti.func
def field_cell_get_point_with_highest_val_1ex(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    cell_3dindex: ti.math.ivec3,
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
) -> ti.math.vec4:
    grid_origin = ti.math.vec3(0.0)
    cell_origin = grid_origin + cell_3dindex * cell_sides_length

    value_max = 0.0
    point_value_max = ti.math.vec3(0.0)

    for i in ti.grouped(
        ti.ndrange(SIDE_SAMPLE_COUNT, SIDE_SAMPLE_COUNT, SIDE_SAMPLE_COUNT)
    ):
        seed_i = ti.cast(
            ti.math.ivec3(
                cell_3dindex.x + SEED,
                cell_3dindex.y + 13 * SEED,
                cell_3dindex.z + 17 * SEED,
            ),
            ti.u32,
        )
        u3d = random.pcg3df(seed_i)
        if ti.static(not JITTER):
            u3d = ti.math.vec3(0.5)

        # Unit cell space
        eval_point = (i + u3d) / SIDE_SAMPLE_COUNT
        # World cell space
        eval_point = eval_point * cell_sides_length + cell_origin

        cos_triphase_i = field_eval(
            normal,
            phi_t,
            phase_t,
            phase_b,
            phase_n,
            eval_point,
            cell_sides_length,
            cos_triperiod,
        )
        cos_triphase_normalized_i = (cos_triphase_i + 1.0) * 0.5
        triphasor3_val = (
            cos_triphase_normalized_i[0]
            + cos_triphase_normalized_i[1]
            + cos_triphase_normalized_i[2]
        ) / 3.0

        if triphasor3_val > value_max:
            value_max = triphasor3_val
            point_value_max = eval_point

    return ti.math.vec4(point_value_max, value_max)


@ti.func
def align_triphase_i_with_j(
    i: math.vec9, j: math.vec9, triperiod: ti.math.vec3
) -> ti.math.vec3:
    p_i = i[:3]
    spherical_basis_i = i[3:6]
    triphase_i = i[6:9]

    p_j = j[:3]
    spherical_basis_j = j[3:6]
    triphase_j = j[6:9]

    cartesian_basis_i = basis3.from_spherical(spherical_basis_i)
    cartesian_basis_j = basis3.from_spherical(spherical_basis_j)

    t_i = cartesian_basis_i[:, 0]
    t_j = cartesian_basis_j[:, 0]
    b_i = cartesian_basis_i[:, 1]
    b_j = cartesian_basis_j[:, 1]
    n_i = cartesian_basis_i[:, 2]
    n_j = cartesian_basis_j[:, 2]

    t_i_sph = direction.cartesian_to_spherical(t_i)
    t_j_sph = direction.cartesian_to_spherical(t_j)
    b_i_sph = direction.cartesian_to_spherical(b_i)
    b_j_sph = direction.cartesian_to_spherical(b_j)
    n_i_sph = direction.cartesian_to_spherical(n_i)
    n_j_sph = direction.cartesian_to_spherical(n_j)

    sine_t_i = math.vec6(p_i, t_i_sph, triperiod[0])
    sine_b_i = math.vec6(p_i, b_i_sph, triperiod[1])
    sine_n_i = math.vec6(p_i, n_i_sph, triperiod[2])

    sine_t_j = math.vec7(p_j, t_j_sph, triphase_j[0], triperiod[0])
    sine_b_j = math.vec7(p_j, b_j_sph, triphase_j[1], triperiod[1])
    sine_n_j = math.vec7(p_j, n_j_sph, triphase_j[2], triperiod[2])

    alignment_measure_t_i_t_j = ti.math.dot(t_i, t_j) ** 2
    alignment_measure_t_i_b_j = ti.math.dot(t_i, b_j) ** 2
    alignment_measure_b_i_b_j = ti.math.dot(b_i, b_j) ** 2
    alignment_measure_n_i_n_j = ti.math.dot(n_i, n_j) ** 2

    phase_t_i_t_j = phasor3.align_phase_i_with_j(sine_t_i, sine_t_j)
    phase_t_i_b_j = phasor3.align_phase_i_with_j(sine_t_i, sine_b_j)
    phase_b_i_b_j = phasor3.align_phase_i_with_j(sine_b_i, sine_b_j)
    phase_n_i_n_j = phasor3.align_phase_i_with_j(sine_n_i, sine_n_j)

    phase_t_i_t_j_complex = direction.polar_to_cartesian(phase_t_i_t_j)
    phase_t_i_b_j_complex = direction.polar_to_cartesian(phase_t_i_b_j)
    phase_b_i_b_j_complex = direction.polar_to_cartesian(phase_b_i_b_j)
    phase_n_i_n_j_complex = direction.polar_to_cartesian(phase_n_i_n_j)

    phase_t_i_complex = (
        alignment_measure_t_i_t_j * phase_t_i_t_j_complex
        + alignment_measure_t_i_b_j * phase_t_i_b_j_complex
    )
    phase_b_i_complex = alignment_measure_b_i_b_j * phase_b_i_b_j_complex
    phase_n_i_complex = alignment_measure_n_i_n_j * phase_n_i_n_j_complex

    # Update the phase of the bitangent of i only if computed phase in complex
    # space is not the null vector

    # Update the phase along the tangent
    if not (ti.abs(phase_t_i_complex) < 0.01).all():
        triphase_i[0] = ti.atan2(phase_t_i_complex.y, phase_t_i_complex.x)
    # Update the phase along the bitangent
    if not (ti.abs(phase_b_i_complex) < 0.01).all():
        triphase_i[1] = ti.atan2(phase_b_i_complex.y, phase_b_i_complex.x)
    # Update the phase along the normal
    if not (ti.abs(phase_n_i_complex) < 0.01).all():
        triphase_i[2] = ti.atan2(phase_n_i_complex.y, phase_n_i_complex.x)

    return triphase_i


@ti.func
def field_eval(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    x: ti.math.vec3,
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
) -> ti.math.vec3:
    origin = ti.math.vec3(0.0)
    i = grid3.cell_3dindex_from_point(x, origin, cell_sides_length)

    sum_weight = 0.0
    sum_weighted_cos = ti.math.vec3(0.0)

    for shifter in ti.grouped(ti.ndrange(3, 3, 3)):
        j = i + shifter - ti.math.ivec3(1, 1, 1)
        j_is_valid = grid3.is_valid_cell_3dindex(j, normal.shape)
        if not j_is_valid:
            continue

        if ti.math.isnan(normal[j][0]):
            continue

        p_j = grid3.cell_center_point(j, origin, cell_sides_length)
        tbn = basis3.from_spherical(ti.math.vec3(normal[j], phi_t[j]))
        t = ti.math.vec3(tbn[0, 0], tbn[1, 0], tbn[2, 0])
        b = ti.math.vec3(tbn[0, 1], tbn[1, 1], tbn[2, 1])
        n = ti.math.vec3(tbn[0, 2], tbn[1, 2], tbn[2, 2])
        triangle_j = ti.math.vec3(0.0, 0.0, 0.0)
        triangle_j[0] = phasor3.eval_angle(x, p_j, t, phase_t[j], cos_triperiod[0])
        triangle_j[1] = phasor3.eval_angle(x, p_j, b, phase_b[j], cos_triperiod[1])
        triangle_j[2] = phasor3.eval_angle(x, p_j, n, phase_n[j], cos_triperiod[2])
        cos_triangle_j = ti.cos(triangle_j)

        w_ij = math.eval_triangle_filter_normalized(
            x,
            p_j,
            cell_sides_length,
        )
        w_ij = w_ij**SPATIAL_WEIGHT_EXPONENT

        sum_weight += w_ij
        sum_weighted_cos += w_ij * cos_triangle_j

    cos_val = sum_weighted_cos / sum_weight

    i_is_valid = grid3.is_valid_cell_3dindex(i, normal.shape)
    if not i_is_valid:
        cos_val = ti.math.nan
    else:
        if ti.math.isnan(normal[i][0]):
            cos_val = ti.math.nan

    return cos_val


@ti.func
def constrain_phase_t(state: ti.u32) -> ti.u32:
    state |= 0b001
    return state


@ti.func
def unconstrain_phase_t(state: ti.u32) -> ti.u32:
    state &= 0b110
    return state


@ti.func
def is_constrained_phase_t(state: ti.u32) -> int:
    mask = 0b001
    masked = mask & state
    return masked == mask


@ti.func
def constrain_phase_b(state: ti.u32) -> ti.u32:
    state |= 0b010
    return state


@ti.func
def unconstrain_phase_b(state: ti.u32) -> ti.u32:
    state &= 0b101
    return state


@ti.func
def is_constrained_phase_b(state: ti.u32) -> int:
    mask = 0b010
    masked = mask & state
    return masked == mask
