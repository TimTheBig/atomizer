import numpy as np
import taichi as ti
from tqdm import tqdm

from . import direction, grid3, transform3


class Field:
    def __init__(self) -> None:
        self.grid: grid3.Grid = None
        self.normal = None
        self.phi_t = None
        self.state = None

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        normal_np = self.normal.to_numpy()
        phi_t_np = self.phi_t.to_numpy()
        state_np = self.state.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["normal"] = normal_np
        dict_array["phi_t"] = phi_t_np
        dict_array["state"] = state_np

        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        shape = self.grid.cell_3dcount

        self.normal = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.phi_t = ti.field(dtype=ti.f32, shape=shape)
        self.state = ti.field(dtype=ti.u32, shape=shape)

        self.normal.from_numpy(dict_array["normal"])
        self.phi_t.from_numpy(dict_array["phi_t"])
        self.state.from_numpy(dict_array["state"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)


class FieldAligner:
    def __init__(self):
        self.normal: list = None
        self.phi_t: list = None
        self.state: list = None
        self.multigrid: grid3.Multigrid = None
        self.iteration_number: int = None

    def allocate_from_field(self, f: Field):
        self.multigrid = grid3.Multigrid()
        self.multigrid.create_from_grid(f.grid)

        self.normal = []
        self.phi_t = []
        self.state = []

        for i in range(self.multigrid.level_count):
            shape_i = self.multigrid.cell_3dcount[i]
            if i != 0:
                normal_i = ti.Vector.field(n=2, dtype=f.normal.dtype, shape=shape_i)
                phi_t0 = ti.field(dtype=f.phi_t.dtype, shape=shape_i)
                state_i = ti.field(dtype=f.state.dtype, shape=shape_i)
            else:
                normal_i = f.normal
                phi_t0 = f.phi_t
                state_i = f.state

            phi_t1 = ti.field(dtype=f.phi_t.dtype, shape=shape_i)

            self.normal.append(normal_i)
            self.phi_t.append([phi_t0, phi_t1])
            self.state.append(state_i)

        self.iteration_number = 0

    def align(self, restrict_func, prolong_func, align_func, alignment_iteration_count):
        level_count_m1 = self.multigrid.level_count - 1
        for level_i in tqdm(range(level_count_m1)):
            self.restrict(restrict_func, level_i)

        for shifter in tqdm(range(self.multigrid.level_count)):
            # level i from self.multigrid.level_count - 1 to 0
            level_i = level_count_m1 - shifter

            if level_i < level_count_m1:
                self.iteration_number = 0
                for _ in range(alignment_iteration_count):
                    self.align_one_level_one_time(align_func, level_i)

            if level_i > 0:
                self.prolong(prolong_func, level_i)

    def restrict(self, restrict_func, level):
        restrict_func(
            self.normal[level],
            self.phi_t[level][0],
            self.state[level],
            self.normal[level + 1],
            self.phi_t[level + 1][0],
            self.state[level + 1],
        )

    def prolong(self, prolong_func, level):
        prolong_func(
            self.normal[level],
            self.phi_t[level][0],
            self.normal[level - 1],
            self.phi_t[level - 1][0],
            self.state[level - 1],
        )

    def align_one_level_one_time(self, align_func, level):
        align_func(
            self.normal[level],
            self.phi_t[level][0],
            self.state[level],
            self.multigrid.cell_sides_length[level],
            self.iteration_number,
            self.phi_t[level][1],
        )
        self.phi_t[level][0], self.phi_t[level][1] = (
            self.phi_t[level][1],
            self.phi_t[level][0],
        )
        self.iteration_number += 1


@ti.func
def tangent_from_normal(n: ti.math.vec3) -> ti.math.vec3:
    """
    https://jcgt.org/published/0006/01/01/
    """
    sign = -1.0
    if n.z >= 0.0:
        sign = 1.0
    a = -1.0 / (sign + n.z)
    b = n.x * n.y * a
    tangent = ti.math.vec3(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x)
    return tangent


@ti.func
def from_spherical(spherical_basis: ti.math.vec3) -> ti.math.mat3:
    normal_sph = ti.math.vec2(spherical_basis[0], spherical_basis[1])
    normal = direction.spherical_to_cartesian(normal_sph)
    tangent_ts = ti.math.vec3(direction.polar_to_cartesian(spherical_basis[2]), 0.0)

    t_from_n = tangent_from_normal(normal)
    b_from_n = ti.math.cross(normal, t_from_n)
    tangent_to_world = transform3.compute_frame_to_canonical_matrix(
        t_from_n, b_from_n, normal, ti.math.vec3(0.0, 0.0, 0.0)
    )
    tangent = transform3.apply_to_vector(tangent_to_world, tangent_ts)
    bitangent = ti.math.cross(normal, tangent)

    return ti.math.mat3([tangent, bitangent, normal]).transpose()
