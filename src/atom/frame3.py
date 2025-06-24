from math import prod

import numpy as np
import taichi as ti

from . import cone, direction, grid3


class Field:
    def __init__(self):
        self.grid = None
        self.point = None
        self.normal = None
        self.phi_t = None

    def compute_memory_usage(self):
        # Return the number of bytes required to store the frame field

        # 6 * 32 bits:
        # - Point: 3 * 32 bits
        # - Spherical normal: 2 * 32 bits
        # - Tangent angle: 1 * 32 bits

        # Unit: bytes (that's why there is * 4)
        return prod(self.grid.cell_3dcount) * 6.0 * 4.0

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        point_np = self.point.to_numpy()
        normal_np = self.normal.to_numpy()
        phi_t_np = self.phi_t.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["point"] = point_np
        dict_array["normal"] = normal_np
        dict_array["phi_t"] = phi_t_np

        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        shape = dict_array["phi_t"].shape

        self.point = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
        self.normal = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.phi_t = ti.field(dtype=ti.f32, shape=shape)

        self.point.from_numpy(dict_array["point"])
        self.normal.from_numpy(dict_array["normal"])
        self.phi_t.from_numpy(dict_array["phi_t"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def to_active_frame_set(self):
        point_np = self.point.to_numpy()
        normal_np = self.normal.to_numpy()
        phi_t_np = self.phi_t.to_numpy()

        point_np = point_np.reshape(-1, 3)
        normal_np = normal_np.reshape(-1, 2)
        phi_t_np = phi_t_np.ravel()

        is_active = ~np.isnan(point_np).any(axis=1)

        point_active = point_np[is_active]
        normal_active = normal_np[is_active]
        phi_t_active = phi_t_np[is_active]

        dict_array = {}
        dict_array["point"] = point_active
        dict_array["normal"] = normal_active
        dict_array["phi_t"] = phi_t_active

        return dict_array

    def save_active_frame_set(self, filename: str):
        dict_array = self.to_active_frame_set()
        np.savez(filename, **dict_array)

        # np.savetxt(
        #     filename,
        #     np.concatenate((point_active, orientation_active.reshape((-1, 1))), axis=1),
        #     fmt="%.3f",
        # )


class Set:
    def __init__(self):
        self.point = None
        self.normal = None
        self.phi_t = None

    def to_numpy(self):
        point_np = self.point.to_numpy()
        normal_np = self.normal.to_numpy()
        phi_t_np = self.phi_t.to_numpy()

        dict_array = {}
        dict_array["point"] = point_np
        dict_array["normal"] = normal_np
        dict_array["phi_t"] = phi_t_np

        return dict_array

    def from_numpy(self, dict_array):
        shape = dict_array["phi_t"].shape

        self.point = ti.Vector.field(n=3, dtype=ti.f32, shape=shape)
        self.normal = ti.Vector.field(n=2, dtype=ti.f32, shape=shape)
        self.phi_t = ti.field(dtype=ti.f32, shape=shape)

        self.point.from_numpy(dict_array["point"])
        self.normal.from_numpy(dict_array["normal"])
        self.phi_t.from_numpy(dict_array["phi_t"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def get_aabb(self):
        set_dict = self.to_numpy()
        point = set_dict["point"]
        p_min = np.min(point, axis=0)
        p_max = np.max(point, axis=0)

        return p_min, p_max

    def get_highest_point(self):
        set_dict = self.to_numpy()
        point = set_dict["point"]
        arg_max = np.argmax(point[:, 2])

        return point[arg_max]


@ti.func
def frame_set_any_point_inside_cone_i_brute_force(
    point: ti.template(),
    normal: ti.template(),
    i: int,
    cone_cos_half_angle: float,
):
    is_unaccessible_i = False
    p_i = point[i]
    n_i = direction.spherical_to_cartesian(normal[i])

    for j in range(point.shape[0]):
        p_j = point[j]
        if ti.math.isnan(p_j).any() or i == j:
            continue

        is_unaccessible_ij = cone.is_point_inside(p_j, p_i, n_i, cone_cos_half_angle)
        if is_unaccessible_ij:
            is_unaccessible_i = True

    return is_unaccessible_i
