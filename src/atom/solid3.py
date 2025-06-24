import numpy as np
import pyvista as pv
import taichi as ti

from . import basis3, direction, grid3, limits, math, phasor3, triphasor3


class BoundaryPointNormal:
    def __init__(self):
        self.point = None
        self.normal = None
        # tuple: (xmin, xmax, ymin, ymax, zmin, zmax)
        self.bounding_box = None

    def create_from_file(self, filepath: str):
        mesh = pv.read(filepath).clean()
        point_np = np.array(mesh.points, dtype=np.float32)
        normal_np = np.array(mesh.point_data["Normals"], dtype=np.float32)
        # rgba = mesh.point_data["RGBA"]
        self.bounding_box = np.array(mesh.bounds)

        pn_count = point_np.shape[0]

        self.point = ti.Vector.field(n=3, dtype=float, shape=pn_count)
        self.normal = ti.Vector.field(n=3, dtype=float, shape=pn_count)

        self.point.from_numpy(point_np)
        self.normal.from_numpy(normal_np)

    def to_numpy(self) -> np.ndarray:
        point_np = self.point.to_numpy()
        normal_np = self.normal.to_numpy()

        dict_array = {}
        dict_array["point"] = point_np
        dict_array["normal"] = normal_np
        dict_array["bounding_box"] = self.bounding_box

        return dict_array

    def get_size(self) -> np.ndarray:
        return np.array(
            [
                self.bounding_box[1] - self.bounding_box[0],
                self.bounding_box[3] - self.bounding_box[2],
                self.bounding_box[5] - self.bounding_box[4],
            ]
        )

    def from_numpy(self, dict_array):
        pn_count = dict_array["point"].shape[0]

        self.point = ti.Vector.field(n=3, dtype=float, shape=pn_count)
        self.normal = ti.Vector.field(n=3, dtype=float, shape=pn_count)
        self.point.from_numpy(dict_array["point"])
        self.normal.from_numpy(dict_array["normal"])
        self.bounding_box = dict_array["bounding_box"]

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)


class SDF:
    def __init__(self) -> None:
        self.grid: grid3.Grid = None
        # Taichi scalar field. dtype: float. shape: 3D
        self.sdf = None

    def compute_memory_usage(self) -> float:
        # Return unit: bytes
        return (
            int(self.sdf.shape[0]) * int(self.sdf.shape[1]) * int(self.sdf.shape[2]) * 4
        )

    def to_numpy(self):
        grid_np = self.grid.to_numpy()
        sdf_np = self.sdf.to_numpy()

        dict_array = {}
        dict_array["grid"] = grid_np
        dict_array["sdf"] = sdf_np
        return dict_array

    def from_numpy(self, dict_array):
        self.grid = grid3.Grid()
        self.grid.from_numpy(dict_array["grid"])

        # Allocate the signed distance field
        self.sdf = ti.field(dtype=ti.f32, shape=self.grid.cell_3dcount)
        self.sdf.from_numpy(dict_array["sdf"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def create_from_bpn(
        self,
        bpn: BoundaryPointNormal,
        cell_sides_length: float,
        bvh=None,
        pow2=True,
    ):

        solid_size = bpn.get_size()

        self.grid = grid3.Grid()
        self.grid.cell_3dcount = np.ceil(solid_size / cell_sides_length).astype(int)
        print(f"SDF cell 3D count: {self.grid.cell_3dcount}")

        if pow2:
            cell_3dcount_max = np.max(self.grid.cell_3dcount)
            cell_count_roundup_power_of_2 = math.roundup_power_of_2(cell_3dcount_max)
            self.grid.cell_3dcount = np.full((3,), cell_count_roundup_power_of_2)
            print(
                f"cell_sides_count, after cubification and roundup to the next power of two: {self.grid.cell_3dcount}"
            )

        self.grid.origin = np.array([0.0, 0.0, 0.0])
        self.grid.cell_sides_length = cell_sides_length

        # Allocate the signed distance field
        self.sdf = ti.field(dtype=ti.f32, shape=self.grid.cell_3dcount)

        if not bvh:
            sdf_create_from_bpn_brute_force(
                bpn.point, bpn.normal, cell_sides_length, self.sdf
            )
        else:
            sdf_create_from_bpn(bpn.point, bpn.normal, bvh, cell_sides_length, self.sdf)

    def init_spherical_direction_field(
        self,
        field: direction.SphericalField,
        init_func,
        all_up: bool = False,
    ):
        field.grid = self.grid

        field.direction = ti.Vector.field(
            n=2, dtype=ti.f32, shape=field.grid.cell_3dcount
        )
        field.state = ti.field(dtype=ti.u32, shape=field.grid.cell_3dcount)

        init_func(
            self.sdf,
            field.grid.cell_sides_length,
            all_up,
            field.direction,
            field.state,
        )

    def init_phasor3_field(
        self, phasor3_field: phasor3.Field, sph_d_field, init_func, all_up
    ):
        phasor3_field.grid = self.grid
        phasor3_field.direction = sph_d_field

        angle_type = ti.f32
        state_type = ti.u32
        phasor3_field.phase = ti.field(
            dtype=angle_type, shape=phasor3_field.grid.cell_3dcount
        )
        phasor3_field.state = ti.field(
            dtype=state_type, shape=phasor3_field.grid.cell_3dcount
        )
        init_func(
            self.sdf,
            phasor3_field.grid.cell_sides_length,
            all_up,
            phasor3_field.phase,
            phasor3_field.state,
        )

    def allocate_basis3_field(
        self,
        normal,
        basis3_field: basis3.Field,
    ):
        basis3_field.grid = self.grid
        basis3_field.normal = normal

        basis3_field.phi_t = ti.field(
            dtype=ti.f32, shape=basis3_field.grid.cell_3dcount
        )
        basis3_field.state = ti.field(
            dtype=ti.u32, shape=basis3_field.grid.cell_3dcount
        )

    def init_triphasor3_field_1ex(
        self,
        init_func,
        basis3_field: basis3.Field,
        phase_n,
        triphasor3_field: triphasor3.Field,
    ):
        triphasor3_field.grid = self.grid
        triphasor3_field.normal = basis3_field.normal
        triphasor3_field.phi_t = basis3_field.phi_t
        triphasor3_field.phase_n = phase_n

        angle_type = ti.f32
        state_type = ti.u32
        triphasor3_field.phase_t = ti.field(
            dtype=angle_type, shape=triphasor3_field.grid.cell_3dcount
        )
        triphasor3_field.phase_b = ti.field(
            dtype=angle_type, shape=triphasor3_field.grid.cell_3dcount
        )
        triphasor3_field.state = ti.field(
            dtype=state_type, shape=triphasor3_field.grid.cell_3dcount
        )
        init_func(
            self.sdf,
            triphasor3_field.normal,
            triphasor3_field.phi_t,
            triphasor3_field.grid.cell_sides_length,
            triphasor3_field.phase_b,
            triphasor3_field.state,
        )


@ti.kernel
def sdf_create_from_bpn_brute_force(
    point: ti.template(),
    normal: ti.template(),
    cell_sides_length: float,
    sdf: ti.template(),
):
    origin = ti.math.vec3(0.0)
    for sdf_i3 in ti.grouped(sdf):
        # Get the cell center
        cell_center = grid3.cell_center_point(sdf_i3, origin, cell_sides_length)

        # Find the closest point
        min_distance = limits.f32_max
        argmin_distance = -1
        for bpn_i1 in range(point.shape[0]):
            distance = ti.math.distance(point[bpn_i1], cell_center)
            if distance < min_distance:
                argmin_distance = bpn_i1
                min_distance = distance

        bpn_closest_normal = normal[argmin_distance]
        # Compute the signed distance
        orientation = ti.math.dot(
            bpn_closest_normal, cell_center - point[argmin_distance]
        )
        if orientation < 0.0:
            min_distance = -min_distance
        sdf[sdf_i3] = min_distance


@ti.kernel
def sdf_create_from_bpn(
    point: ti.template(),
    normal: ti.template(),
    bvh: ti.template(),
    cell_sides_length: float,
    sdf: ti.template(),
):
    origin = ti.math.vec3(0.0)
    min_distance = limits.f32_max
    for sdf_i3 in ti.grouped(sdf):
        # Get the cell center
        cell_center = grid3.cell_center_point(sdf_i3, origin, cell_sides_length)

        # Get point and normal from bvh
        idx = bvh.nearest_neighbor(cell_center)
        bpn_closest_point = point[idx]
        bpn_closest_normal = normal[idx]

        # Compute unsigned distance
        min_distance = ti.math.length(cell_center - bpn_closest_point)

        # Compute the signed distance
        orientation = ti.math.dot(bpn_closest_normal, cell_center - bpn_closest_point)
        if orientation < 0.0:
            min_distance = -min_distance
        sdf[sdf_i3] = min_distance


@ti.func
def sdf_compute_gradient_central(
    sdf: ti.template(), xyz: ti.math.ivec3, cell_sides_length: float
) -> ti.math.vec3:
    """Assume xyz is valid"""
    xm1yz = xyz + ti.math.ivec3(-1, 0, 0)
    xp1yz = xyz + ti.math.ivec3(1, 0, 0)
    xym1z = xyz + ti.math.ivec3(0, -1, 0)
    xyp1z = xyz + ti.math.ivec3(0, 1, 0)
    xyzm1 = xyz + ti.math.ivec3(0, 0, -1)
    xyzp1 = xyz + ti.math.ivec3(0, 0, 1)

    xm1yz_is_valid = grid3.is_valid_cell_3dindex(xm1yz, sdf.shape)
    xp1yz_is_valid = grid3.is_valid_cell_3dindex(xp1yz, sdf.shape)
    xym1z_is_valid = grid3.is_valid_cell_3dindex(xym1z, sdf.shape)
    xyp1z_is_valid = grid3.is_valid_cell_3dindex(xyp1z, sdf.shape)
    xyzm1_is_valid = grid3.is_valid_cell_3dindex(xyzm1, sdf.shape)
    xyzp1_is_valid = grid3.is_valid_cell_3dindex(xyzp1, sdf.shape)

    dx = 2.0 * cell_sides_length
    if not (xm1yz_is_valid and xp1yz_is_valid):
        dx = cell_sides_length
    dy = 2.0 * cell_sides_length
    if not (xym1z_is_valid and xyp1z_is_valid):
        dy = cell_sides_length
    dz = 2.0 * cell_sides_length
    if not (xyzm1_is_valid and xyzp1_is_valid):
        dz = cell_sides_length

    ix1 = xp1yz
    ix0 = xm1yz
    iy1 = xyp1z
    iy0 = xym1z
    iz1 = xyzp1
    iz0 = xyzm1

    if not xp1yz_is_valid:
        ix1 = xyz
    if not xm1yz_is_valid:
        ix0 = xyz
    if not xyp1z_is_valid:
        iy1 = xyz
    if not xym1z_is_valid:
        iy0 = xyz
    if not xyzp1_is_valid:
        iz1 = xyz
    if not xyzm1_is_valid:
        iz0 = xyz

    dfdx = (sdf[ix1] - sdf[ix0]) / dx
    dfdy = (sdf[iy1] - sdf[iy0]) / dy
    dfdz = (sdf[iz1] - sdf[iz0]) / dz

    return ti.math.vec3(dfdx, dfdy, dfdz)


@ti.func
def sdf_compute_closest_normal_central(
    sdf: ti.template(), xyz: ti.math.ivec3, cell_sides_length: float
) -> ti.math.vec3:
    grad = sdf_compute_gradient_central(sdf, xyz, cell_sides_length)
    return math.normalize_safe(grad)


@ti.func
def sdf_compute_closest_curvature_central(
    sdf: ti.template(), xyz: ti.math.ivec3, cell_sides_length: float
):
    xm1yz = xyz + ti.math.ivec3(-1, 0, 0)
    xym1z = xyz + ti.math.ivec3(0, -1, 0)
    xyzm1 = xyz + ti.math.ivec3(0, 0, -1)
    xp1yz = xyz + ti.math.ivec3(1, 0, 0)
    xyp1z = xyz + ti.math.ivec3(0, 1, 0)
    xyzp1 = xyz + ti.math.ivec3(0, 0, 1)

    xm1yz_is_valid = grid3.is_valid_cell_3dindex(xm1yz, sdf.shape)
    xym1z_is_valid = grid3.is_valid_cell_3dindex(xym1z, sdf.shape)
    xyzm1_is_valid = grid3.is_valid_cell_3dindex(xyzm1, sdf.shape)
    xp1yz_is_valid = grid3.is_valid_cell_3dindex(xp1yz, sdf.shape)
    xyp1z_is_valid = grid3.is_valid_cell_3dindex(xyp1z, sdf.shape)
    xyzp1_is_valid = grid3.is_valid_cell_3dindex(xyzp1, sdf.shape)

    ix1 = xp1yz
    ix0 = xm1yz
    iy1 = xyp1z
    iy0 = xym1z
    iz1 = xyzp1
    iz0 = xyzm1

    if not xp1yz_is_valid:
        ix1 = xyz
    if not xm1yz_is_valid:
        ix0 = xyz
    if not xyp1z_is_valid:
        iy1 = xyz
    if not xym1z_is_valid:
        iy0 = xyz
    if not xyzp1_is_valid:
        iz1 = xyz
    if not xyzm1_is_valid:
        iz0 = xyz

    dndx = direction.diff_normalized(
        sdf_compute_closest_normal_central(sdf, ix0, cell_sides_length),
        sdf_compute_closest_normal_central(sdf, ix1, cell_sides_length),
    )
    dndy = direction.diff_normalized(
        sdf_compute_closest_normal_central(sdf, iy0, cell_sides_length),
        sdf_compute_closest_normal_central(sdf, iy1, cell_sides_length),
    )
    dndz = direction.diff_normalized(
        sdf_compute_closest_normal_central(sdf, iz0, cell_sides_length),
        sdf_compute_closest_normal_central(sdf, iz1, cell_sides_length),
    )

    return (dndx + dndy + dndz) / 3.0
