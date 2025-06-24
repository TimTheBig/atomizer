from math import prod

import numpy as np
import taichi as ti

from . import (
    basis3,
    color,
    direction,
    fff3,
    frame3,
    grid2,
    grid3,
    limits,
    math,
    phasor3,
    solid3,
    toolpath3,
    transform3,
    triphasor3,
)


class BoundaryPointNormalDrawer:
    def __init__(self):
        self.line_vertex = None
        self.per_vertex_color = None

    def init_from_bpn(self, bpn: solid3.BoundaryPointNormal, normal_scale: float):
        line_vertex_count = bpn.point.shape[0] * 2
        self.line_vertex = ti.Vector.field(n=3, dtype=ti.f32, shape=line_vertex_count)
        self.per_vertex_color = ti.Vector.field(
            n=3, dtype=ti.f32, shape=line_vertex_count
        )
        boundary_point_normal_drawer_init_lines(
            bpn.point, bpn.normal, normal_scale, self.line_vertex
        )
        boundary_point_normal_drawer_init_per_vertex_color(self.per_vertex_color)


class GridMesh2:
    """
    A 2D grid mesh embedded in 3D.
    """

    def __init__(self) -> None:
        # Number of subdivisions along the X and Y axes
        self.subdivisions: np.ndarray = None

        # The size of all the sides.
        # Assume all sides have the same size.
        self.size: float = None
        # Origin of the mesh
        self.origin = None
        # Orientation of the mesh
        self.orientation = None

        # Vertices of the mesh
        self.vertex = None
        # Indices of the mesh
        self.index = None
        # Normals of the mesh
        self.normal = None
        # Per vertex color
        self.per_vertex_color = None

    def create(self, subdivisions: np.ndarray):
        self.subdivisions = subdivisions

        self.size = 1.0
        self.origin = np.array([0.0, 0.0, 0.0])
        self.orientation = np.array([0.0, 0.0])

        # Memory allocation
        vertex_count = (subdivisions[0] + 1) * (subdivisions[1] + 1)
        # Per surface element, we have two triangles, implying we have 6
        # indices per surface subdivision
        index_count = subdivisions[0] * subdivisions[1] * 6
        self.vertex = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.normal = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.per_vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.index = ti.field(dtype=ti.i32, shape=index_count)

        grid_mesh2_init_indices(subdivisions, self.index)
        grid_mesh2_update_vertex_normal(
            self.origin,
            self.orientation,
            self.size,
            self.subdivisions,
            self.vertex,
            self.normal,
        )

    def update_vertex_normal(self):
        grid_mesh2_update_vertex_normal(
            self.origin,
            self.orientation,
            self.size,
            self.subdivisions,
            self.vertex,
            self.normal,
        )

    def update_per_vertex_color_with_sdf(self, sdf: solid3.SDF):
        min_max = math.eval_min_max(sdf.sdf)
        grid_mesh2_update_per_vertex_color_with_sdf(
            self.vertex,
            sdf.sdf,
            min_max,
            sdf.grid.cell_sides_length,
            self.per_vertex_color,
        )

    def update_per_vertex_color_with_frame_set_region(
        self, frame_set: frame3.Set, triperiod: np.ndarray, atom_index: int
    ):
        grid_mesh2_update_per_vertex_color_with_frame_set_region(
            self.vertex,
            frame_set.point,
            frame_set.normal,
            frame_set.phi_t,
            triperiod,
            atom_index,
            self.per_vertex_color,
        )

    def update_per_vertex_color_with_triphasor3_multigrid_1ex(
        self,
        multigrid: triphasor3.MultigridAligner,
        sdf,
        level: int,
        linear=False,
        draw_grid=False,
    ):
        if not linear:
            grid_mesh2_update_per_vertex_color_with_triphasor3_field_nearest(
                self.vertex,
                multigrid.normal[level],
                multigrid.phi_t[level],
                multigrid.phase_t[level][0],
                multigrid.phase_b[level][0],
                multigrid.phase_n[level],
                multigrid.state[level],
                multigrid.multigrid.cell_sides_length[level],
                self.per_vertex_color,
            )
        else:
            grid_mesh2_update_per_vertex_color_with_triphasor3_field_linear_1ex(
                sdf,
                self.vertex,
                multigrid.normal[level],
                multigrid.phi_t[level],
                multigrid.phase_t[level][0],
                multigrid.phase_b[level][0],
                multigrid.phase_n[level],
                multigrid.state[level],
                multigrid.multigrid.cell_sides_length[level],
                multigrid.cos_triperiod,
                draw_grid,
                self.per_vertex_color,
            )

    def update_per_vertex_color_with_triphasor3_field_1ex(
        self,
        triphasor3_field: triphasor3.Field,
        linear=False,
        draw_grid=False,
    ):
        cos_triperiod = fff3.cos_triperiod_1ex_from_cell_sides_length_kernel(
            triphasor3_field.grid.cell_sides_length
        )
        if not linear:
            grid_mesh2_update_per_vertex_color_with_triphasor3_field_nearest(
                self.vertex,
                triphasor3_field.normal,
                triphasor3_field.phi_t,
                triphasor3_field.phase_t,
                triphasor3_field.phase_b,
                triphasor3_field.phase_n,
                triphasor3_field.state,
                triphasor3_field.grid.cell_sides_length,
                self.per_vertex_color,
            )
        else:
            grid_mesh2_update_per_vertex_color_with_triphasor3_field_linear_1ex(
                self.vertex,
                triphasor3_field.normal,
                triphasor3_field.phi_t,
                triphasor3_field.phase_t,
                triphasor3_field.phase_b,
                triphasor3_field.phase_n,
                triphasor3_field.state,
                triphasor3_field.grid.cell_sides_length,
                cos_triperiod,
                draw_grid,
                self.per_vertex_color,
            )

    def update_per_vertex_color_with_phasor3(
        self,
        phasor3_field: phasor3.Field,
        cos_period: float,
        linear=False,
        draw_grid=False,
    ):
        if not linear:
            grid_mesh2_update_per_vertex_color_with_phasor3_field_nearest(
                self.vertex,
                phasor3_field.direction,
                phasor3_field.phase,
                phasor3_field.state,
                phasor3_field.grid.cell_sides_length,
                draw_grid,
                self.per_vertex_color,
            )
        else:
            grid_mesh2_update_per_vertex_color_with_phasor3_field_linear_1ex(
                self.vertex,
                phasor3_field.direction,
                phasor3_field.phase,
                phasor3_field.state,
                phasor3_field.grid.cell_sides_length,
                cos_period,
                draw_grid,
                self.per_vertex_color,
            )


class ConeMesh:
    def __init__(self):
        self.base_vertex_count = None
        self.opening_angle = None

        self.size: float = None
        # Origin of the mesh
        self.origin = None
        # Orientation of the mesh
        self.orientation = None

        # Vertices of the mesh
        self.vertex = None
        # Indices of the mesh
        self.index = None
        # Normals of the mesh
        self.normal = None

    def create(self, base_vertex_count, opening_angle, size, origin, orientation):
        self.base_vertex_count = base_vertex_count
        self.opening_angle = opening_angle

        self.size = size
        self.origin = origin
        self.orientation = orientation

        # Memory allocation
        vertex_count = base_vertex_count + 1
        index_count = base_vertex_count * 3
        self.vertex = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.normal = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.index = ti.field(dtype=ti.i32, shape=index_count)

        cone_mesh_init_indices(base_vertex_count, self.index)
        cone_mesh_update_vertex_normal(
            self.origin,
            self.orientation,
            self.size,
            self.opening_angle,
            self.base_vertex_count,
            self.vertex,
            self.normal,
        )

    def update_vertex_normal(self):
        cone_mesh_update_vertex_normal(
            self.origin,
            self.orientation,
            self.size,
            self.opening_angle,
            self.base_vertex_count,
            self.vertex,
            self.normal,
        )


class SDFDrawer:
    def __init__(self) -> None:
        self.vertex = None
        self.per_vertex_color = None
        self.radius = None

    def init_from_sdf(self, sdf: solid3.SDF):
        # XY plane
        vertex_count = prod(sdf.grid.cell_3dcount[:2])
        self.vertex = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.per_vertex_color = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count)
        self.radius = sdf.grid.cell_sides_length * 0.3

    def update(
        self, sdf: solid3.SDF, cell_3dindex_to_view: np.ndarray, only_see_one_cell: bool
    ):
        min_max = math.eval_min_max(sdf.sdf)
        sdf_drawer_update(
            sdf.sdf,
            cell_3dindex_to_view,
            min_max,
            sdf.grid.cell_sides_length,
            only_see_one_cell,
            self.vertex,
            self.per_vertex_color,
        )


class SphericalDirectionFieldDrawer:
    def __init__(self) -> None:
        self.line_vertex = None
        self.per_vertex_color = None

    def init_from_field(self, field: direction.SphericalField):
        # XY plane
        line_vertex_count = prod(field.grid.cell_3dcount[:2]) * 2
        self.line_vertex = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=line_vertex_count,
        )
        self.per_vertex_color = ti.Vector.field(
            n=3, dtype=ti.f32, shape=line_vertex_count
        )
        spherical_direction_field_drawer_init_per_vertex_color(self.per_vertex_color)

    def update(
        self,
        field: direction.SphericalField,
        cell_3dindex_to_view: np.ndarray,
        only_see_one_cell: bool,
    ):
        spherical_direction_field_drawer_update_lines(
            field.direction,
            field.state,
            cell_3dindex_to_view,
            field.grid.cell_sides_length,
            only_see_one_cell,
            self.line_vertex,
        )


class BasisFieldDrawer:
    def __init__(self) -> None:
        self.line_vertex = None
        self.per_vertex_color = None

    def init_from_field(self, field: basis3.Field):
        line_vertex_count = prod(field.grid.cell_3dcount[:2]) * 4
        self.line_vertex = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=line_vertex_count,
        )
        self.per_vertex_color = ti.Vector.field(
            n=3, dtype=ti.f32, shape=line_vertex_count
        )
        basis3_field_drawer_init_per_vertex_color(self.per_vertex_color)

    def update(
        self,
        field: basis3.Field,
        cell_3dindex_to_view: np.ndarray,
        only_see_one_cell: bool,
    ):
        basis3_field_drawer_update_lines(
            field.normal,
            field.phi_t,
            field.state,
            cell_3dindex_to_view,
            field.grid.cell_sides_length,
            only_see_one_cell,
            self.line_vertex,
        )


class FrameFieldDrawer:
    def __init__(self):
        self.points = None
        self.lines = None

    def init_from_field(self, frame3_field: frame3.Field):
        shape_xy = prod(frame3_field.grid.cell_3dcount[:2])
        self.points = ti.Vector.field(3, ti.f32, shape=shape_xy)
        line_vertex_count = shape_xy * 4
        self.lines = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=line_vertex_count,
        )

    def update(
        self,
        frame3_field: frame3.Field,
        cell_3dindex_to_view: np.ndarray,
        only_see_one_cell: bool,
    ):
        frame3_field_drawer_update_point_line(
            frame3_field.point,
            frame3_field.normal,
            frame3_field.phi_t,
            cell_3dindex_to_view,
            frame3_field.grid.cell_sides_length,
            only_see_one_cell,
            self.points,
            self.lines,
        )


class FrameSetDrawer:
    def __init__(self):
        self.point = None
        self.line_vertex = None
        self.line_per_vertex_color = None

    def init_from_field(self, frame3_set: frame3.Set):
        shape = frame3_set.phi_t.shape[0]
        self.point = ti.Vector.field(3, ti.f32, shape=shape)
        line_vertex_count = shape * 4
        self.line_vertex = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=line_vertex_count,
        )
        self.line_per_vertex_color = ti.Vector.field(
            n=3,
            dtype=ti.f32,
            shape=line_vertex_count,
        )
        basis3_field_drawer_init_per_vertex_color(self.line_per_vertex_color)

    def update(
        self,
        frame3_set: frame3.Set,
        index_min_max: np.ndarray,
        scale_factor: float,
    ):
        frame3_set_drawer_update_point_line(
            frame3_set.point,
            frame3_set.normal,
            frame3_set.phi_t,
            index_min_max,
            scale_factor,
            self.point,
            self.line_vertex,
        )


class FrameDataSetDrawer:
    def __init__(self):
        self.point_color = None

    def allocate(self, frame_data_set: toolpath3.FrameDataSet):
        scalar_count = frame_data_set.cost.shape[0]
        self.point_color = ti.Vector.field(n=3, dtype=ti.f32, shape=scalar_count)

    def draw_non_supporting(self, frame_data_set: toolpath3.FrameDataSet):
        frame_data_set_drawer_draw_non_supporting_frames(
            frame_data_set.state, self.point_color
        )

    def draw_atoms_in_a_cycle(self, frame_data_set: toolpath3.FrameDataSet):
        frame_data_set_drawer_draw_atoms_in_a_cycle(
            frame_data_set.state, self.point_color
        )

    def draw_atoms_any_below(self, frame_data_set: toolpath3.FrameDataSet):
        frame_data_set_drawer_draw_any_below(frame_data_set.state, self.point_color)

    def draw_wall(self, frame_data_set: toolpath3.FrameDataSet):
        frame_data_set_drawer_draw_wall(frame_data_set.state, self.point_color)

    def draw_unaccessible(self, frame_data_set: toolpath3.FrameDataSet):
        frame_data_set_drawer_draw_unaccessible(frame_data_set.state, self.point_color)

    def draw_cost(self, frame_data_set: toolpath3.FrameDataSet):
        min_max = np.array([0.0, 5.0])
        frame_data_set_drawer_draw_scalar_field(
            frame_data_set.cost, min_max, self.point_color
        )


class ToolpathDrawer:
    def __init__(self):
        self.vertex = None
        self.line_vertex = None
        self.per_vertex_color = None

    def allocate(self, point_count: int):
        self.vertex = ti.Vector.field(3, ti.f32, shape=point_count)
        line_vertex_count = point_count * 2
        self.line_vertex = ti.Vector.field(n=3, dtype=ti.f32, shape=line_vertex_count)
        self.per_vertex_color = ti.Vector.field(
            n=3, dtype=ti.f32, shape=line_vertex_count
        )

    def update(
        self,
        toolpath: toolpath3.Toolpath,
        start_index: int,
        end_index: int,
        hide_travels: bool = False,
    ):
        toolpath_drawer_update_vertices(
            toolpath.point,
            toolpath.point_count,
            toolpath.travel_type,
            start_index,
            end_index,
            hide_travels,
            self.vertex,
        )
        toolpath_drawer_update_lines(
            toolpath.point_count,
            toolpath.travel_type,
            toolpath.length_from_start,
            hide_travels,
            self.vertex,
            self.line_vertex,
            self.per_vertex_color,
        )


class BasisMultigridDrawer:
    def __init__(self):
        self.line_vertex = None
        self.per_vertex_color = None

    def init_from_multigrid(self, multigrid: basis3.FieldAligner):
        # XY plane
        self.line_vertex = []
        self.per_vertex_color = []
        for i in range(multigrid.multigrid.level_count):
            line_vertex_count_i = (
                multigrid.multigrid.cell_3dcount[i][0]
                * multigrid.multigrid.cell_3dcount[i][1]
                * 4
            )
            line_vertices_i = ti.Vector.field(
                n=3,
                dtype=ti.f32,
                shape=line_vertex_count_i,
            )
            per_vertex_colors_i = ti.Vector.field(
                n=3, dtype=ti.f32, shape=line_vertex_count_i
            )
            basis3_field_drawer_init_per_vertex_color(per_vertex_colors_i)
            self.line_vertex.append(line_vertices_i)
            self.per_vertex_color.append(per_vertex_colors_i)

    def update(
        self,
        multigrid: basis3.FieldAligner,
        cell_3dindex_to_view: np.ndarray,
        level: int,
        only_see_one_cell: bool,
    ):
        basis3_field_drawer_update_lines(
            multigrid.normal[level],
            multigrid.phi_t[level][0],
            multigrid.state[level],
            cell_3dindex_to_view,
            multigrid.multigrid.cell_sides_length[level],
            only_see_one_cell,
            self.line_vertex[level],
        )


class Triphasor3MultigridDrawer:
    def __init__(self) -> None:
        self.vertex = None
        self.per_vertex_color = None
        self.radius = None

    def init_from_multigrid(self, multigrid: triphasor3.MultigridAligner):
        # XY plane
        self.vertex = []
        self.per_vertex_color = []
        self.radius = []
        for i in range(multigrid.multigrid.level_count):
            vertex_count_i = (
                multigrid.multigrid.cell_3dcount[i][0]
                * multigrid.multigrid.cell_3dcount[i][1]
            )
            vertices_i = ti.Vector.field(n=3, dtype=ti.f32, shape=vertex_count_i)
            per_vertex_colors_i = ti.Vector.field(
                n=3, dtype=ti.f32, shape=vertex_count_i
            )
            radius_i = multigrid.multigrid.cell_sides_length[i] * 0.3

            self.vertex.append(vertices_i)
            self.per_vertex_color.append(per_vertex_colors_i)
            self.radius.append(radius_i)

    def update(
        self,
        multigrid: triphasor3.MultigridAligner,
        cell_3dindex_to_view: np.ndarray,
        level: int,
        only_see_one_cell: bool,
    ):
        triphasor3_drawer_update(
            multigrid.normal[level],
            multigrid.phi_t[level],
            multigrid.phase_t[level][0],
            multigrid.phase_b[level][0],
            multigrid.phase_n[level],
            multigrid.state[level],
            cell_3dindex_to_view,
            multigrid.multigrid.cell_sides_length[level],
            only_see_one_cell,
            self.vertex[level],
            self.per_vertex_color[level],
        )


@ti.kernel
def toolpath_drawer_update_vertices(
    point: ti.types.ndarray(),
    point_count: int,
    travel_type: ti.types.ndarray(),
    start_index: int,
    end_index: int,
    hide_travels: int,
    vertex: ti.template(),
):
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for i in range(point_count):
        ip1_safe = ti.math.min(i + 1, point_count - 1)
        is_travel = (
            not travel_type[i] == toolpath3.TRAVEL_TYPE_DEPOSITION
            and not travel_type[ip1_safe] == toolpath3.TRAVEL_TYPE_DEPOSITION
        )
        if not hide_travels:
            is_travel = False

        vertex[i] = ti.math.vec3(ti.math.nan)

        if i >= start_index and i < end_index and not is_travel:
            p_i = ti.math.vec3(point[i, 0], point[i, 1], point[i, 2])
            vertex[i] = transform3.apply_to_point(rotate_x, p_i)


@ti.kernel
def toolpath_drawer_update_lines(
    point_count: int,
    travel_type: ti.types.ndarray(),
    length_from_start: ti.types.ndarray(),
    hide_travels: int,
    vertex: ti.template(),
    line_vertex: ti.template(),
    per_vertex_color: ti.template(),
):
    total_length = length_from_start[point_count - 1]
    for i in range(point_count):
        ip1_safe = ti.math.min(i + 1, point_count - 1)
        is_travel = (
            not travel_type[i] == toolpath3.TRAVEL_TYPE_DEPOSITION
            and not travel_type[ip1_safe] == toolpath3.TRAVEL_TYPE_DEPOSITION
        )
        if not hide_travels:
            is_travel = False

        line_vertex[i * 2 + 0] = ti.math.vec3(ti.math.nan)
        line_vertex[i * 2 + 1] = ti.math.vec3(ti.math.nan)

        if i > 0:
            if not is_travel:
                line_vertex[i * 2 + 0] = vertex[i - 1]
                line_vertex[i * 2 + 1] = vertex[i]

            if travel_type[i] == toolpath3.TRAVEL_TYPE_DEPOSITION:
                # per_vertex_color[i * 2 + 0] = color.class3dark21
                # per_vertex_color[i * 2 + 1] = color.class3dark21
                per_vertex_color[i * 2 + 0] = color.turbo(
                    length_from_start[i] / total_length
                )
                per_vertex_color[i * 2 + 1] = color.turbo(
                    length_from_start[i] / total_length
                )
            else:
                per_vertex_color[i * 2 + 0] = color.class3dark20
                per_vertex_color[i * 2 + 1] = color.class3dark20


@ti.kernel
def frame_data_set_drawer_draw_non_supporting_frames(
    state: ti.template(), point_color: ti.template()
):
    for i in state:
        point_color[i] = color.class3dark20
        if toolpath3.atom_is_not_supporting(state[i]):
            point_color[i] = color.class3dark21


@ti.kernel
def frame_data_set_drawer_draw_atoms_in_a_cycle(
    state: ti.template(), point_color: ti.template()
):
    for i in state:
        point_color[i] = color.class3dark20
        if toolpath3.atom_is_in_cycle(state[i]):
            point_color[i] = color.class3dark21


@ti.kernel
def frame_data_set_drawer_draw_any_below(
    state: ti.template(), point_color: ti.template()
):
    for i in state:
        point_color[i] = color.class3dark20
        if toolpath3.atom_any_below(state[i]):
            point_color[i] = color.class3dark21


@ti.kernel
def frame_data_set_drawer_draw_wall(state: ti.template(), point_color: ti.template()):
    for i in state:
        point_color[i] = color.class3dark20
        if toolpath3.atom_is_wall(state[i]):
            point_color[i] = color.class3dark21


@ti.kernel
def frame_data_set_drawer_draw_unaccessible(
    state: ti.template(), point_color: ti.template()
):
    for i in state:
        point_color[i] = color.class3dark20
        if toolpath3.atom_is_not_supporting(state[i]):
            point_color[i] = color.class3set21
        if toolpath3.atom_is_unaccessible(state[i]):
            point_color[i] = color.class3dark21


@ti.kernel
def frame_data_set_drawer_draw_scalar_field(
    scalar: ti.template(), min_max: ti.math.vec2, point_color: ti.template()
):
    for i in scalar:
        scalar_normalized_i = (scalar[i] - min_max[0]) / (min_max[1] - min_max[0])
        point_color[i] = color.turbo(scalar_normalized_i)


@ti.kernel
def spherical_direction_field_drawer_update_lines(
    field_direction: ti.template(),
    field_state: ti.template(),
    cell_3dindex_to_view: ti.math.ivec3,
    field_cell_sides_length: float,
    only_see_one_cell: bool,
    drawer_lines: ti.template(),
):
    slice_number = cell_3dindex_to_view[2]
    field_origin = ti.math.vec3(0.0)
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    # For each direction in the 3D grid, init the corresponding line
    for cell_2dindex in ti.grouped(
        ti.ndrange(field_state.shape[0], field_state.shape[1])
    ):
        cell_3dindex = ti.math.ivec3(cell_2dindex[0], cell_2dindex[1], slice_number)

        angle_i = field_direction[cell_3dindex]
        dir_i = direction.spherical_to_cartesian(angle_i)

        drawer_1dindex = grid2.cell_1dindex_from_2dindex(
            cell_2dindex, field_state.shape[0]
        )

        # Get the cell center
        cell_center_i = grid3.cell_center_point(
            cell_3dindex,
            field_origin,
            field_cell_sides_length,
        )

        cell_center_i = transform3.apply_to_point(rotate_x, cell_center_i)
        dir_i = transform3.apply_to_vector(rotate_x, dir_i)

        if ti.math.isnan(field_direction[cell_3dindex].x) or (
            only_see_one_cell and not (cell_3dindex == cell_3dindex_to_view).all()
        ):
            drawer_lines[drawer_1dindex * 2 + 0] = ti.math.nan
            drawer_lines[drawer_1dindex * 2 + 1] = ti.math.nan
        else:
            drawer_lines[drawer_1dindex * 2 + 0] = cell_center_i
            drawer_lines[drawer_1dindex * 2 + 1] = (
                cell_center_i + dir_i * field_cell_sides_length * 0.4
            )
        if direction.is_constrained(field_state[cell_3dindex]):
            # drawer_lines[drawer_1dindex * 2 + 0] = cell_center_i
            drawer_lines[drawer_1dindex * 2 + 1] = (
                cell_center_i + dir_i * 0.5 * field_cell_sides_length * 0.4
            )


@ti.kernel
def basis3_field_drawer_update_lines(
    field_normal: ti.template(),
    field_phi_t: ti.template(),
    field_state: ti.template(),
    cell_3dindex_to_view: ti.math.ivec3,
    field_cell_sides_length: float,
    only_see_one_cell: bool,
    drawer_lines: ti.template(),
):
    slice_number = cell_3dindex_to_view[2]
    field_origin = ti.math.vec3(0.0)
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    # For each direction in the 3D grid, init the corresponding line
    for cell_2dindex in ti.grouped(
        ti.ndrange(field_state.shape[0], field_state.shape[1])
    ):
        cell_3dindex = ti.math.ivec3(cell_2dindex[0], cell_2dindex[1], slice_number)
        drawer_1dindex = grid2.cell_1dindex_from_2dindex(
            cell_2dindex, field_state.shape[0]
        )

        sph_normal_i = field_normal[cell_3dindex]
        normal_i = direction.spherical_to_cartesian(sph_normal_i)
        # ts: tangent_space
        tangent_i_ts = ti.math.vec3(
            direction.polar_to_cartesian(field_phi_t[cell_3dindex]), 0.0
        )

        t_from_n = basis3.tangent_from_normal(normal_i)
        b_from_n = ti.math.cross(normal_i, t_from_n)
        tangent_to_world = transform3.compute_frame_to_canonical_matrix(
            t_from_n, b_from_n, normal_i, ti.math.vec3(0.0, 0.0, 0.0)
        )
        # tangent_i = transform3.apply_to_point(tangent_to_world, tangent_i_ts)
        tangent_i = transform3.apply_to_vector(tangent_to_world, tangent_i_ts)

        # Get the cell center
        cell_center_i = grid3.cell_center_point(
            cell_3dindex,
            field_origin,
            field_cell_sides_length,
        )

        cell_center_i = transform3.apply_to_point(rotate_x, cell_center_i)
        normal_i = transform3.apply_to_vector(rotate_x, normal_i)
        tangent_i = transform3.apply_to_vector(rotate_x, tangent_i)

        scale_factor = 0.4
        if fff3.tangent_boundary_is_constrained(field_state[cell_3dindex]):
            scale_factor = 0.1

        if ti.math.isnan(field_normal[cell_3dindex].x) or (
            only_see_one_cell and not (cell_3dindex == cell_3dindex_to_view).all()
        ):
            drawer_lines[drawer_1dindex * 4 + 0] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 1] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 2] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 3] = ti.math.nan
        else:
            drawer_lines[drawer_1dindex * 4 + 0] = cell_center_i
            drawer_lines[drawer_1dindex * 4 + 1] = (
                cell_center_i + normal_i * field_cell_sides_length * scale_factor
            )
            drawer_lines[drawer_1dindex * 4 + 2] = (
                cell_center_i - tangent_i * field_cell_sides_length * scale_factor
            )
            drawer_lines[drawer_1dindex * 4 + 3] = (
                cell_center_i + tangent_i * field_cell_sides_length * scale_factor
            )


@ti.kernel
def frame3_field_drawer_update_point_line(
    field_point: ti.template(),
    field_normal: ti.template(),
    field_phi_t: ti.template(),
    cell_3dindex_to_view: ti.math.ivec3,
    field_cell_sides_length: float,
    only_see_one_cell: bool,
    drawer_points: ti.template(),
    drawer_lines: ti.template(),
):
    layer_height = fff3.layer_height_from_cell_sides_length(field_cell_sides_length)
    slice_number = cell_3dindex_to_view[2]
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for cell_2dindex in ti.grouped(
        ti.ndrange(field_point.shape[0], field_point.shape[1])
    ):
        cell_3dindex = ti.math.ivec3(cell_2dindex[0], cell_2dindex[1], slice_number)
        drawer_1dindex = grid2.cell_1dindex_from_2dindex(
            cell_2dindex, field_point.shape[0]
        )

        sph_normal_i = field_normal[cell_3dindex]
        normal_i = direction.spherical_to_cartesian(sph_normal_i)
        # ts: tangent_space
        tangent_i_ts = ti.math.vec3(
            direction.polar_to_cartesian(field_phi_t[cell_3dindex]), 0.0
        )

        t_from_n = basis3.tangent_from_normal(normal_i)
        b_from_n = ti.math.cross(normal_i, t_from_n)
        tangent_to_world = transform3.compute_frame_to_canonical_matrix(
            t_from_n, b_from_n, normal_i, ti.math.vec3(0.0, 0.0, 0.0)
        )
        # tangent_i = transform3.apply_to_point(tangent_to_world, tangent_i_ts)
        tangent_i = transform3.apply_to_vector(tangent_to_world, tangent_i_ts)

        p_i = field_point[cell_3dindex]

        p_i = transform3.apply_to_point(rotate_x, p_i)
        normal_i = transform3.apply_to_vector(rotate_x, normal_i)
        tangent_i = transform3.apply_to_vector(rotate_x, tangent_i)

        drawer_points[drawer_1dindex] = p_i

        scale_factor = 0.5

        if ti.math.isnan(field_normal[cell_3dindex].x) or (
            only_see_one_cell and not (cell_3dindex == cell_3dindex_to_view).all()
        ):
            drawer_lines[drawer_1dindex * 4 + 0] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 1] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 2] = ti.math.nan
            drawer_lines[drawer_1dindex * 4 + 3] = ti.math.nan
        else:
            drawer_lines[drawer_1dindex * 4 + 0] = p_i
            drawer_lines[drawer_1dindex * 4 + 1] = (
                p_i + normal_i * layer_height * scale_factor
            )
            drawer_lines[drawer_1dindex * 4 + 2] = (
                p_i - tangent_i * layer_height * scale_factor
            )
            drawer_lines[drawer_1dindex * 4 + 3] = (
                p_i + tangent_i * layer_height * scale_factor
            )


@ti.kernel
def frame3_set_drawer_update_point_line(
    set_point: ti.template(),
    set_normal: ti.template(),
    set_phi_t: ti.template(),
    index_min_max: ti.math.ivec2,
    scale_factor: float,
    drawer_point: ti.template(),
    drawer_line: ti.template(),
):
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for i in set_point:
        normal_i = direction.spherical_to_cartesian(set_normal[i])
        # ts: tangent_space
        tangent_i_ts = ti.math.vec3(direction.polar_to_cartesian(set_phi_t[i]), 0.0)

        t_from_n = basis3.tangent_from_normal(normal_i)
        b_from_n = ti.math.cross(normal_i, t_from_n)
        tangent_to_world = transform3.compute_frame_to_canonical_matrix(
            t_from_n, b_from_n, normal_i, ti.math.vec3(0.0, 0.0, 0.0)
        )
        tangent_i = transform3.apply_to_vector(tangent_to_world, tangent_i_ts)

        p_i = set_point[i]

        p_i = transform3.apply_to_point(rotate_x, p_i)
        normal_i = transform3.apply_to_vector(rotate_x, normal_i)
        tangent_i = transform3.apply_to_vector(rotate_x, tangent_i)

        drawer_point[i] = p_i

        if (
            (ti.math.isnan(set_point[i])).any()
            or i < index_min_max[0]
            or i > index_min_max[1]
        ):
            drawer_point[i] = ti.math.nan
            drawer_line[i * 4 + 0] = ti.math.nan
            drawer_line[i * 4 + 1] = ti.math.nan
            drawer_line[i * 4 + 2] = ti.math.nan
            drawer_line[i * 4 + 3] = ti.math.nan
        else:
            drawer_line[i * 4 + 0] = p_i
            drawer_line[i * 4 + 1] = p_i + normal_i * scale_factor
            drawer_line[i * 4 + 2] = p_i - tangent_i * scale_factor
            drawer_line[i * 4 + 3] = p_i + tangent_i * scale_factor


@ti.kernel
def sdf_drawer_update(
    sdf: ti.template(),
    cell_3dindex_to_view: ti.math.ivec3,
    sdf_min_max: ti.math.vec2,
    cell_sides_length: float,
    only_see_one_cell: bool,
    drawer_vertex: ti.template(),
    drawer_per_vertex_color: ti.template(),
):
    slice_number = cell_3dindex_to_view[2]
    field_origin = ti.math.vec3(0.0)
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for i2 in ti.grouped(ti.ndrange(sdf.shape[0], sdf.shape[1])):
        i1 = grid2.cell_1dindex_from_2dindex(i2, sdf.shape[0])
        i3 = ti.math.ivec3(i2[0], i2[1], slice_number)

        # Vertex
        p_i = grid3.cell_center_point(i3, field_origin, cell_sides_length)
        drawer_vertex[i1] = transform3.apply_to_point(rotate_x, p_i)

        # Color

        sd_normalized = (sdf[i3] - sdf_min_max[0]) / (sdf_min_max[1] - sdf_min_max[0])
        color_i = color.viridis(sd_normalized)

        drawer_per_vertex_color[i1] = color_i

        if (i3 == cell_3dindex_to_view).all():
            drawer_per_vertex_color[i1] = color.class3set21
        else:
            if only_see_one_cell:
                drawer_vertex[i1] = ti.math.vec3(ti.math.nan)

        if ti.math.isnan(sdf[i3]) or sdf[i3] > 0.0:
            drawer_per_vertex_color[i1] = color.class3set20
        if sdf[i3] == limits.f32_max:
            drawer_per_vertex_color[i1] = color.class3dark21


@ti.kernel
def triphasor3_drawer_update(
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    state: ti.template(),
    cell_3dindex_to_view: ti.math.ivec3,
    cell_sides_length: float,
    only_see_one_cell: bool,
    drawer_vertex: ti.template(),
    drawer_per_vertex_color: ti.template(),
):
    slice_number = cell_3dindex_to_view[2]
    field_origin = ti.math.vec3(0.0)
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for i2 in ti.grouped(ti.ndrange(phase_t.shape[0], phase_t.shape[1])):
        i3 = ti.math.ivec3(i2[0], i2[1], slice_number)

        vertex_i = grid3.cell_center_point(i3, field_origin, cell_sides_length)
        if ti.math.isnan(normal[i3][0]):
            vertex_i = ti.math.vec3(ti.math.nan)

        cos_phase_normalized_i = (ti.cos(phase_b[i3]) + 1.0) * 0.5
        per_vertex_color_i = color.viridis(cos_phase_normalized_i)
        # if triphasor3.is_constrained_phase_b(state[i3]):
        #     per_vertex_color_i = color.class3dark21

        drawer_1dindex = grid2.cell_1dindex_from_2dindex(i2, phase_t.shape[0])

        vertex_i = transform3.apply_to_point(rotate_x, vertex_i)
        drawer_vertex[drawer_1dindex] = vertex_i
        drawer_per_vertex_color[drawer_1dindex] = per_vertex_color_i

        if not (i3 == cell_3dindex_to_view).all() and only_see_one_cell:
            drawer_vertex[drawer_1dindex] = ti.math.vec3(ti.math.nan)


@ti.kernel
def spherical_direction_field_drawer_init_per_vertex_color(
    per_vertex_color: ti.template(),
):
    for i in range(per_vertex_color.shape[0] // 2):
        per_vertex_color[i * 2 + 0] = ti.math.vec3(68.0, 1.0, 84.0) / 255.0
        per_vertex_color[i * 2 + 1] = ti.math.vec3(253.0, 231.0, 37.0) / 255.0


@ti.kernel
def basis3_field_drawer_init_per_vertex_color(
    per_vertex_color: ti.template(),
):
    for i in range(per_vertex_color.shape[0] // 4):
        per_vertex_color[i * 4 + 0] = ti.math.vec3(68.0, 1.0, 84.0) / 255.0
        per_vertex_color[i * 4 + 1] = ti.math.vec3(253.0, 231.0, 37.0) / 255.0
        per_vertex_color[i * 4 + 2] = color.class3dark21
        per_vertex_color[i * 4 + 3] = color.class3dark21


@ti.kernel
def boundary_point_normal_drawer_init_lines(
    point: ti.template(),
    normal: ti.template(),
    normal_scale: float,
    line_vertex: ti.template(),
):
    """
    In: p, n, normal_scale
    Out: l
    """
    rotate_x = transform3.rotate_x(-ti.math.pi * 0.5)
    for i in range(point.shape[0]):
        p_i_t = transform3.apply_to_point(rotate_x, point[i])
        n_i_t = transform3.apply_to_vector(rotate_x, normal[i])
        line_vertex[i * 2] = p_i_t
        line_vertex[i * 2 + 1] = p_i_t + n_i_t * normal_scale


@ti.kernel
def boundary_point_normal_drawer_init_per_vertex_color(
    per_vertex_color: ti.template(),
):
    for i in range(per_vertex_color.shape[0] // 2):
        per_vertex_color[i * 2] = ti.math.vec3(68.0, 1.0, 84.0) / 255.0
        per_vertex_color[i * 2 + 1] = ti.math.vec3(253.0, 231.0, 37.0) / 255.0


@ti.kernel
def grid_mesh2_init_indices(subdivisions: ti.math.ivec2, index: ti.template()):
    for i in range(subdivisions[1]):
        for j in range(subdivisions[0]):
            index_0 = i * (subdivisions[0] + 1) + j
            index_1 = index_0 + 1
            index_2 = (i + 1) * (subdivisions[0] + 1) + j
            index_3 = index_2 + 1

            index[(i * subdivisions[0] + j) * 6 + 0] = index_0
            index[(i * subdivisions[0] + j) * 6 + 1] = index_2
            index[(i * subdivisions[0] + j) * 6 + 2] = index_1
            index[(i * subdivisions[0] + j) * 6 + 3] = index_1
            index[(i * subdivisions[0] + j) * 6 + 4] = index_2
            index[(i * subdivisions[0] + j) * 6 + 5] = index_3


@ti.kernel
def cone_mesh_init_indices(base_vertex_count: int, index: ti.template()):
    for i in range(base_vertex_count):
        next_index = (i + 1) % base_vertex_count
        index[3 * i] = i
        index[3 * i + 1] = next_index
        index[3 * i + 2] = base_vertex_count  # Apex index


@ti.kernel
def cone_mesh_update_vertex_normal(
    origin: ti.math.vec3,
    orientation: ti.math.vec2,
    size: float,
    opening_angle: float,
    base_vertex_count: int,
    vertex: ti.template(),
    normal: ti.template(),
):
    theta = orientation[0]
    phi = orientation[1]
    S = transform3.scale_uniformly(size)
    R_y = transform3.rotate_y(theta)
    R_z = transform3.rotate_z(phi)
    R_x = transform3.rotate_x(-ti.math.pi * 0.5)
    R_x_2 = transform3.rotate_x(-ti.math.pi * 0.5)
    T = transform3.translate(origin)
    T_0p5 = transform3.translate(ti.math.vec3(0.0, -1.0, 0.0))
    M = R_x @ T @ R_z @ R_y @ R_x_2 @ S @ T_0p5

    radius = math.atan(opening_angle * 0.5)
    height = 1.0

    angle_increment = 2.0 * ti.math.pi / base_vertex_count

    for i in range(base_vertex_count):
        angle = i * angle_increment
        x = radius * ti.cos(angle)
        z = radius * ti.sin(angle)
        vertex[i] = ti.math.vec3(x, 0.0, z)

        # Calculate normal for the base vertex
        to_apex = ti.math.vec3(-x, height, -z)
        normal[i] = -ti.math.normalize(to_apex)

        vertex[i] = transform3.apply_to_point(M, vertex[i])
        normal[i] = transform3.apply_to_vector(M, normal[i])

    # Initialize apex vertex and normal
    vertex[base_vertex_count] = ti.math.vec3(0.0, height, 0.0)
    normal[base_vertex_count] = ti.math.vec3(0.0, 1.0, 0.0)

    vertex[base_vertex_count] = transform3.apply_to_point(M, vertex[base_vertex_count])
    normal[base_vertex_count] = transform3.apply_to_vector(M, normal[base_vertex_count])


@ti.kernel
def grid_mesh2_update_vertex_normal(
    origin: ti.math.vec3,
    orientation: ti.math.vec2,
    size: float,
    subdivisions: ti.math.ivec2,
    vertex: ti.template(),
    normal: ti.template(),
):
    theta = orientation[0]
    phi = orientation[1]
    S = transform3.scale_uniformly(size)
    R_y = transform3.rotate_y(theta)
    R_z = transform3.rotate_z(phi)
    R_x = transform3.rotate_x(-ti.math.pi * 0.5)
    T = transform3.translate(origin)
    M = R_x @ T @ R_z @ R_y @ S

    for i in range(subdivisions[1] + 1):
        for j in range(subdivisions[0] + 1):
            index1 = i * (subdivisions[0] + 1) + j
            p_ij = ti.math.vec3(
                j / subdivisions[0] - 0.5, i / subdivisions[1] - 0.5, 0.0
            )
            n_ij = ti.math.vec3(0.0, 0.0, 0.99)

            p_ij = transform3.apply_to_point(M, p_ij)
            n_ij = transform3.apply_to_vector(M, n_ij)

            vertex[index1] = p_ij
            normal[index1] = n_ij


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_sdf(
    point: ti.template(),
    sdf: ti.template(),
    sdf_min_max: ti.math.vec2,
    sdf_cell_sides_length: float,
    per_vertex_color: ti.template(),
):
    sdf_min = sdf_min_max[0]
    sdf_max = sdf_min_max[1]
    sdf_origin = ti.math.vec3(0.0)

    R_x = transform3.rotate_x(ti.math.pi * 0.5)

    for i in point:
        p_i = transform3.apply_to_point(R_x, point[i])
        sdf_cell_3dindex = grid3.cell_3dindex_from_point(
            p_i, sdf_origin, sdf_cell_sides_length
        )
        is_valid_sdf_cell = grid3.is_valid_cell_3dindex(sdf_cell_3dindex, sdf.shape)
        signed_distance = sdf_min
        if is_valid_sdf_cell:
            signed_distance = sdf[sdf_cell_3dindex]
            boundary_width = fff3.layer_height_from_cell_sides_length(
                sdf_cell_sides_length
            )
            if fff3.is_boundary_region(signed_distance, boundary_width):
                signed_distance = sdf_max
        signed_distance_normalized = (signed_distance - sdf_min) / (sdf_max - sdf_min)
        per_vertex_color[i] = color.turbo(signed_distance_normalized)


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_frame_set_region(
    grid_point: ti.template(),
    frame_point: ti.template(),
    frame_normal: ti.template(),
    frame_phi_t: ti.template(),
    triperiod: ti.math.vec3,
    atom_index: int,
    per_vertex_color: ti.template(),
):
    R_x = transform3.rotate_x(ti.math.pi * 0.5)

    for i in grid_point:
        p_i = transform3.apply_to_point(R_x, grid_point[i])

        atom = math.vec6(
            frame_point[atom_index], frame_normal[atom_index], frame_phi_t[atom_index]
        )

        region_number_i = toolpath3.atom_get_region_number(p_i, atom, triperiod)

        color_i = ti.math.vec3(0.5, 0.5, 0.5)
        if region_number_i == 0:
            color_i = ti.math.vec3(color.class3set20)
        elif region_number_i == 1:
            color_i = ti.math.vec3(color.class3set22)
        elif region_number_i == 2:
            color_i = ti.math.vec3(color.class3set22)
        elif region_number_i == 3:
            color_i = ti.math.vec3(color.class3dark20)
        elif region_number_i == 4:
            color_i = ti.math.vec3(color.class3dark21)
        elif region_number_i == 5:
            color_i = ti.math.vec3(color.class3dark22)

        per_vertex_color[i] = color_i


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_phasor3_field_nearest(
    point: ti.template(),
    dir: ti.template(),
    phase: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    draw_grid: bool,
    per_vertex_color: ti.template(),
):
    origin = ti.math.vec3(0.0)
    R_x = transform3.rotate_x(ti.math.pi * 0.5)
    for point_index in point:
        x = transform3.apply_to_point(R_x, point[point_index])
        cell_3dindex = grid3.cell_3dindex_from_point(x, origin, cell_sides_length)
        is_valid_cell = grid3.is_valid_cell_3dindex(cell_3dindex, dir.shape)

        if is_valid_cell and not ti.math.isnan(dir[cell_3dindex][0]):
            # cos_phase_normalized_i = (ti.cos(phase[cell_3dindex]) + 1.0) * 0.5
            # per_vertex_color[point_index] = color.viridis(cos_phase_normalized_i)

            phase_normalized = direction.cartesian_to_polar(
                direction.polar_to_cartesian(phase[cell_3dindex])
            )
            phase_normalized = (phase_normalized + ti.math.pi) / (2.0 * ti.math.pi)
            per_vertex_color[point_index] = color.viridis(phase_normalized)
        else:
            per_vertex_color[point_index] = color.class3dark21

        # Draw grid
        if draw_grid:
            edge_width_grid_space = 0.05
            edge_width_over_2 = edge_width_grid_space * cell_sides_length * 0.5
            p_i_grid_space = (x - origin + edge_width_over_2) / cell_sides_length
            p_i_cell_space = ti.math.fract(p_i_grid_space)
            if (p_i_cell_space < edge_width_grid_space).any():
                per_vertex_color[point_index] = ti.math.vec3(0.1)


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_triphasor3_field_nearest(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    per_vertex_color: ti.template(),
):
    origin = ti.math.vec3(0.0)
    R_x = transform3.rotate_x(ti.math.pi * 0.5)
    for i in point:
        p_i = transform3.apply_to_point(R_x, point[i])
        cell_3dindex = grid3.cell_3dindex_from_point(p_i, origin, cell_sides_length)
        is_valid_cell = grid3.is_valid_cell_3dindex(cell_3dindex, normal.shape)

        # DEBUG
        # if not (i == ti.math.ivec3(140, 140, 140)).all():
        #     continue

        if is_valid_cell and not ti.math.isnan(normal[cell_3dindex][0]):
            cos_phase_normalized_i = (ti.cos(phase_n[cell_3dindex]) + 1.0) * 0.5
            per_vertex_color[i] = color.viridis(cos_phase_normalized_i)
        else:
            per_vertex_color[i] = color.class3dark21

        # Draw grid
        # edge_width_grid_space = 0.05
        # edge_width_over_2 = edge_width_grid_space * cell_sides_length * 0.5
        # p_i_grid_space = (p_i - origin + edge_width_over_2) / cell_sides_length
        # p_i_cell_space = ti.math.fract(p_i_grid_space)
        # if (p_i_cell_space < edge_width_grid_space).any():
        #     per_vertex_color[i] = ti.math.vec3(0.1)


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_phasor3_field_linear_1ex(
    point: ti.template(),
    direction: ti.template(),
    phase: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    cos_period: float,
    draw_grid: bool,
    per_vertex_color: ti.template(),
):
    origin = ti.math.vec3(0.0)
    R_x = transform3.rotate_x(ti.math.pi * 0.5)
    for point_index in point:
        x = transform3.apply_to_point(R_x, point[point_index])
        i = grid3.cell_3dindex_from_point(x, origin, cell_sides_length)
        i_is_valid = grid3.is_valid_cell_3dindex(i, direction.shape)

        if i_is_valid and not ti.math.isnan(direction[i][0]):
            cos_phase_i = phasor3.field_eval(
                direction, phase, x, cell_sides_length, cos_period
            )
            cos_phase_normalized_i = (cos_phase_i + 1.0) * 0.5
            per_vertex_color[point_index] = color.viridis(cos_phase_normalized_i)
            # Draw layers
            # if cos_phase_normalized_i > 0.9:
            #     per_vertex_color[point_index] = color.class3dark21
            if phasor3.is_phase_constrained(state[i]):
                per_vertex_color[point_index] = (
                    0.5 * ti.math.vec3(color.class3dark21)
                    + 0.5 * per_vertex_color[point_index]
                )
        else:
            per_vertex_color[point_index] = color.class3set20

        # Draw grid
        if draw_grid:
            edge_width_grid_space = 0.05
            edge_width_over_2 = edge_width_grid_space * cell_sides_length * 0.5
            p_i_grid_space = (x - origin + edge_width_over_2) / cell_sides_length
            p_i_cell_space = ti.math.fract(p_i_grid_space)
            if (p_i_cell_space < edge_width_grid_space).any():
                per_vertex_color[point_index] = ti.math.vec3(0.1)


@ti.kernel
def grid_mesh2_update_per_vertex_color_with_triphasor3_field_linear_1ex(
    point: ti.template(),
    normal: ti.template(),
    phi_t: ti.template(),
    phase_t: ti.template(),
    phase_b: ti.template(),
    phase_n: ti.template(),
    state: ti.template(),
    cell_sides_length: float,
    cos_triperiod: ti.math.vec3,
    draw_grid: bool,
    per_vertex_color: ti.template(),
):
    origin = ti.math.vec3(0.0)
    R_x = transform3.rotate_x(ti.math.pi * 0.5)
    nozzle_width = fff3.deposition_width_from_cell_sides_length(cell_sides_length)
    layer_height = fff3.layer_height_from_cell_sides_length(cell_sides_length)
    for point_index in point:
        x = transform3.apply_to_point(R_x, point[point_index])
        i = grid3.cell_3dindex_from_point(x, origin, cell_sides_length)
        i_is_valid = grid3.is_valid_cell_3dindex(i, normal.shape)

        if i_is_valid:
            if ti.math.isnan(normal[i][0]):
                per_vertex_color[point_index] = color.class3set20
                continue
            cos_triphase_i = triphasor3.field_eval(
                normal,
                phi_t,
                phase_t,
                phase_b,
                phase_n,
                x,
                cell_sides_length,
                cos_triperiod,
            )

            # CAUTION: How the triphasor3 field is evaluated here should be synchronized with
            # how it is evalated in triphasor3.field_cell_get_point_with_highest_val

            # attuation_factor_tn = ti.math.smoothstep(0.0, layer_height * 0.25, -sdf[i])
            # attuation_factor_b = 1.0
            # if wall_sdf != limits.f32_max:
            #     attuation_factor_b = ti.math.smoothstep(
            #         0.0, nozzle_width * 0.25, wall_sdf[i]
            #     )
            # attuation_factor = ti.min(attuation_factor_tn, attuation_factor_b)

            # cos_triphase_normalized_i = ti.abs(cos_triphase_i)
            cos_triphase_normalized_i = (cos_triphase_i + 1.0) * 0.5
            triphasor3_val = (
                cos_triphase_normalized_i[0]
                + cos_triphase_normalized_i[1]
                + cos_triphase_normalized_i[2]
            ) / 3.0
            # triphasor3_val = (
            #     cos_triphase_normalized_i[1] + cos_triphase_normalized_i[2]
            # ) / 2.0
            # triphasor3_val = cos_triphase_normalized_i[0]
            per_vertex_color[point_index] = color.viridis(triphasor3_val)

            # cos_triphase_normalized_i = ti.abs(cos_triphase_i)
            # per_vertex_color[point_index] = color.viridis(cos_triphase_normalized_i[1])

            # cos_triphase_normalized_i = (cos_triphase_i + 1.0) * 0.5
            # per_vertex_color[point_index] = color.viridis(cos_triphase_normalized_i[1])

            # if triphasor3.is_constrained_phase_b(state[i]):
            #     per_vertex_color[point_index] *= ti.math.vec3(1.0, 0.0, 0.0)
        else:
            per_vertex_color[point_index] = color.class3set20

        # Draw grid
        if draw_grid:
            edge_width_grid_space = 0.05
            edge_width_over_2 = edge_width_grid_space * cell_sides_length * 0.5
            p_i_grid_space = (x - origin + edge_width_over_2) / cell_sides_length
            p_i_cell_space = ti.math.fract(p_i_grid_space)
            if (p_i_cell_space < edge_width_grid_space).any():
                per_vertex_color[point_index] = ti.math.vec3(0.1)
