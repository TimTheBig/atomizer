import numpy as np
import taichi as ti

import atom.basis3
import atom.color
import atom.drawer3
import atom.fff3
import atom.frame3
import atom.grid3
import atom.limits
import atom.math
import atom.phasor3
import atom.solid3
import atom.transform3
import atom.triphasor3

ti.init(arch=ti.gpu, debug=False, offline_cache_cleaning_policy="never")


@ti.kernel
def basis_init_vertices_per_vertex_color(
    vertices: ti.template(), per_vertex_color: ti.template()
):
    vertices[0] = ti.math.vec3(1.0, 0.0, 0.0)
    vertices[1] = ti.math.vec3(-1.0, 0.0, 0.0)
    vertices[2] = ti.math.vec3(0.0, 1.0, 0.0)
    vertices[3] = ti.math.vec3(0.0, -1.0, 0.0)
    vertices[4] = ti.math.vec3(0.0, 0.0, 1.0)
    vertices[5] = ti.math.vec3(0.0, 0.0, -1.0)

    rotate_x = atom.transform3.rotate_x(-ti.math.pi * 0.5)
    for i in range(6):
        vertices[i] = atom.transform3.apply_to_point(rotate_x, vertices[i])

    # x axis is greener toward positive values
    per_vertex_color[0] = ti.math.vec3(atom.color.class3dark20)
    per_vertex_color[1] = ti.math.vec3(1.0)
    # y axis is more orange toward positive values
    per_vertex_color[2] = ti.math.vec3(atom.color.class3dark21)
    per_vertex_color[3] = ti.math.vec3(1.0)
    # z axis is more purple toward positive values
    per_vertex_color[4] = ti.math.vec3(atom.color.class3dark22)
    per_vertex_color[5] = ti.math.vec3(1.0)


@ti.kernel
def box_init_vertices_per_vertex_color(
    vertices: ti.template(), per_vertex_color: ti.template()
):
    # Loop over axes
    for axis_index in ti.static(range(3)):
        origin = ti.math.vec3(-1.0)
        for line_index in ti.static(range(4)):
            grid_for_extrusion = ti.math.ivec3(2)
            grid_for_extrusion[axis_index] = 1
            origin_shift = atom.grid3.cell_3dindex_from_1dindex(
                line_index, grid_for_extrusion
            )
            origin_shifted = origin + origin_shift * 2.0
            for point_index in ti.static(range(2)):
                color_i = ti.math.vec3(1.0)
                if point_index == 1:
                    # Extrude along current axis
                    translation = ti.math.vec3(0.0)
                    translation[axis_index] = 2.0
                    origin_shifted += translation

                    color_i = ti.math.vec3(atom.color.class3dark2[axis_index])

                vertices[point_index + line_index * 2 + axis_index * 8] = origin_shifted
                per_vertex_color[point_index + line_index * 2 + axis_index * 8] = (
                    color_i
                )


@ti.kernel
def align_biangle_i_with_j_kernel(
    triphasor3_i: atom.math.vec6,
    triphasor3_j: atom.math.vec6,
    triperiod: ti.math.vec3,
    grid3: atom.math.vec7,
    particle_center: ti.template(),
    particle_color: ti.template(),
) -> ti.math.vec3:
    triphasor3_point_i = ti.math.vec3(-0.5, 0.0, 0.0)
    triphasor3_point_j = ti.math.vec3(0.5, 0.0, 0.0)
    triphasor3_spherical_basis_i = triphasor3_i[:3]
    triphasor3_spherical_basis_j = triphasor3_j[:3]
    triphasor3_triphase_i = triphasor3_i[3:6]
    triphasor3_triphase_j = triphasor3_j[3:6]

    sine3_i = atom.math.vec9(
        triphasor3_point_i, triphasor3_spherical_basis_i, triphasor3_triphase_i
    )
    sine3_j = atom.math.vec9(
        triphasor3_point_j, triphasor3_spherical_basis_j, triphasor3_triphase_j
    )

    triphasor3_triphase_i = atom.triphasor3.align_triphase_i_with_j(
        sine3_i, sine3_j, triperiod
    )

    # Unpack grid3
    grid3_cell_3dcount = ti.cast(grid3[:3], int)
    grid3_origin = grid3[3:6]
    grid3_cell_sides_length = grid3[6]

    rotate_x = atom.transform3.rotate_x(-ti.math.pi * 0.5)

    for cell_3dindex in ti.grouped(
        ti.ndrange(grid3_cell_3dcount[0], grid3_cell_3dcount[1], grid3_cell_3dcount[2])
    ):
        cell_1dindex = atom.grid3.cell_1dindex_from_3dindex(
            cell_3dindex, ti.math.ivec2(grid3_cell_3dcount[0], grid3_cell_3dcount[1])
        )
        cell_center = atom.grid3.cell_center_point(
            cell_3dindex, grid3_origin, grid3_cell_sides_length
        )
        cell_center = atom.transform3.apply_to_point(rotate_x, cell_center)

        triphasor3_point = triphasor3_point_j
        triphasor3_spherical_basis = triphasor3_spherical_basis_j
        triphasor3_triphase = triphasor3_triphase_j
        if cell_center.x > 0.0:
            triphasor3_point = triphasor3_point_i
            triphasor3_spherical_basis = triphasor3_spherical_basis_i
            triphasor3_triphase = triphasor3_triphase_i

        tbn = atom.basis3.from_spherical(triphasor3_spherical_basis)
        t = ti.math.vec3(tbn[0, 0], tbn[1, 0], tbn[2, 0])
        b = ti.math.vec3(tbn[0, 1], tbn[1, 1], tbn[2, 1])
        n = ti.math.vec3(tbn[0, 2], tbn[1, 2], tbn[2, 2])
        t = atom.transform3.apply_to_point(rotate_x, t)
        b = atom.transform3.apply_to_point(rotate_x, b)
        n = atom.transform3.apply_to_point(rotate_x, n)

        triangle = ti.math.vec3(0.0, 0.0, 0.0)
        triangle[0] = atom.phasor3.eval_angle(
            cell_center, triphasor3_point, t, triphasor3_triphase[0], triperiod[0]
        )
        triangle[1] = atom.phasor3.eval_angle(
            cell_center, triphasor3_point, b, triphasor3_triphase[1], triperiod[1]
        )
        triangle[2] = atom.phasor3.eval_angle(
            cell_center, triphasor3_point, n, triphasor3_triphase[2], triperiod[2]
        )

        for i in range(3):
            triangle[i] = ti.math.atan2(ti.sin(triangle[i]), ti.cos(triangle[i]))

        cos_triangle = ti.cos(triangle)
        cos_triangle_sum = cos_triangle[0] + cos_triangle[1] + cos_triangle[2]
        cos_triangle_sum_normalized = (cos_triangle_sum + 3.0) / 6.0

        threshold_vertex = 0.9
        threshold_edge = 0.3
        is_vertex = cos_triangle_sum_normalized > threshold_vertex
        is_tangent_edge = (
            ti.abs(triangle[1]) < threshold_edge
            and ti.abs(triangle[2]) < threshold_edge
        )
        is_bitangent_edge = (
            ti.abs(triangle[0]) < threshold_edge
            and ti.abs(triangle[2]) < threshold_edge
        )
        is_normal_edge = (
            ti.abs(triangle[0]) < threshold_edge
            and ti.abs(triangle[1]) < threshold_edge
        )
        if (
            not is_vertex
            and not is_tangent_edge
            and not is_bitangent_edge
            and not is_normal_edge
        ):
            cell_center = ti.math.vec3(atom.limits.f32_max)

        particle_color[cell_1dindex] = atom.color.viridis(cos_triangle_sum_normalized)
        particle_center[cell_1dindex] = cell_center

        if is_tangent_edge:
            particle_color[cell_1dindex] = atom.color.class3dark20
        if is_bitangent_edge:
            particle_color[cell_1dindex] = atom.color.class3dark21
        if is_normal_edge:
            particle_color[cell_1dindex] = atom.color.class3dark22

    return triphasor3_triphase_i


def experiment_align_triangle_i_with_j():
    theta_n_i = 0.0
    phi_n_i = 0.0
    phi_t_i = 0.0
    theta_n_j = 0.0
    phi_n_j = 0.0
    phi_t_j = 0.0

    triphase_i = ti.math.vec3(0.0)
    phase_t_j = 0.0
    phase_b_j = 0.0
    phase_n_j = 0.0

    period_t = 1.0
    period_b = 1.0
    period_n = 1.0

    # Visu
    grid3_sides_length = 2.0
    grid3_cell_3dcount = ti.math.ivec3(64)
    grid3_origin = ti.math.vec3(-1.0)
    grid3_cell_sides_length = grid3_sides_length / grid3_cell_3dcount[0]
    # Pack grid3 parameters
    grid3 = atom.math.vec7(grid3_cell_3dcount, grid3_origin, grid3_cell_sides_length)

    # Particle centers and colors initialization
    particle_count = grid3_cell_3dcount.x * grid3_cell_3dcount.y * grid3_cell_3dcount.z
    particle_center = ti.Vector.field(n=3, dtype=float, shape=(particle_count,))
    particle_color = ti.Vector.field(n=3, dtype=float, shape=(particle_count,))

    basis_vertices = ti.Vector.field(3, dtype=ti.f32, shape=6)
    basis_per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=6)
    basis_init_vertices_per_vertex_color(basis_vertices, basis_per_vertex_color)

    box_vertices = ti.Vector.field(3, dtype=ti.f32, shape=24)
    box_per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=24)
    box_init_vertices_per_vertex_color(box_vertices, box_per_vertex_color)

    window = ti.ui.Window("Triphasor3", (768, 768))
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 4.0)

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.5) as w:
            theta_n_i = w.slider_float("theta_n_i", theta_n_i, 0.0, ti.math.pi)
            theta_n_j = w.slider_float("theta_n_j", theta_n_j, 0.0, ti.math.pi)
            phi_n_i = w.slider_float(
                "phi_n_i",
                phi_n_i,
                -ti.math.pi,
                ti.math.pi,
            )
            phi_n_j = w.slider_float(
                "phi_n_j",
                phi_n_j,
                -ti.math.pi,
                ti.math.pi,
            )
            phi_t_i = w.slider_float("phi_t_i", phi_t_i, -ti.math.pi, ti.math.pi)
            phi_t_j = w.slider_float("phi_t_j", phi_t_j, -ti.math.pi, ti.math.pi)
            phase_t_j = w.slider_float("phase_t_j", phase_t_j, -ti.math.pi, ti.math.pi)
            phase_b_j = w.slider_float("phase_b_j", phase_b_j, -ti.math.pi, ti.math.pi)
            phase_n_j = w.slider_float("phase_n_j", phase_n_j, -ti.math.pi, ti.math.pi)
            period_t = w.slider_float("period_t", period_t, 0.5, 2.0)
            period_b = w.slider_float("period_b", period_b, 0.5, 2.0)
            period_n = w.slider_float("period_n", period_n, 0.5, 2.0)

        spherical_basis_i = ti.math.vec3(theta_n_i, phi_n_i, phi_t_i)
        spherical_basis_j = ti.math.vec3(theta_n_j, phi_n_j, phi_t_j)
        triphase_j = ti.math.vec3(phase_t_j, phase_b_j, phase_n_j)
        triperiod = ti.math.vec3(period_t, period_b, period_n)

        sine_i = atom.math.vec6(spherical_basis_i, triphase_i)
        sine_j = atom.math.vec6(spherical_basis_j, triphase_j)
        triphase_i = align_biangle_i_with_j_kernel(
            sine_i, sine_j, triperiod, grid3, particle_center, particle_color
        )

        camera.track_user_inputs(window, movement_speed=0.04, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.particles(
            centers=particle_center,
            radius=grid3_cell_sides_length * 0.5,
            per_vertex_color=particle_color,
        )
        scene.lines(basis_vertices, width=5, per_vertex_color=basis_per_vertex_color)
        scene.lines(box_vertices, width=5, per_vertex_color=box_per_vertex_color)

        canvas.scene(scene)
        window.show()


def experiment_triphasor3_extract_maxima_1ex():
    solid_name = "triangle_24"
    bpn_path = f"data/point_normal/{solid_name}.npz"
    # sdf_infill_path = f"data/sdf/{solid_name}_infill.npz"
    sdf_infill_path = f"data/sdf/{solid_name}.npz"
    triphasor_path = f"data/triphasor/{solid_name}.npz"

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)
    print(f"Domain size: {bpn.get_size()}")
    print(f"bounding box [xmin, xmax, ymin, ymax, zmin, zmax]: {bpn.bounding_box}")
    print(f"Point count: {bpn.point.shape[0]}")

    sdf_infill = atom.solid3.SDF()
    sdf_infill.load(sdf_infill_path)
    print(f"Sides cell count: {sdf_infill.grid.cell_3dcount[0]}")
    layer_height = atom.fff3.layer_height_from_cell_sides_length_kernel(
        sdf_infill.grid.cell_sides_length
    )
    print(f"Layer height: {layer_height}")
    cos_triperiod = atom.fff3.cos_triperiod_1ex_from_cell_sides_length_kernel(
        sdf_infill.grid.cell_sides_length
    )

    triphasor3_field = atom.triphasor3.Field()
    triphasor3_field.load(triphasor_path)

    frame_field = atom.frame3.Field()
    triphasor3_field_cell_point_val = triphasor3_field.allocate_frame_field(frame_field)
    triphasor3_field.extract_frame_field(
        frame_field, triphasor3_field_cell_point_val, cos_triperiod, 1
    )
    atom.fff3.frame_field_filter_point_too_close_to_boundary(
        sdf_infill.sdf, sdf_infill.grid.cell_sides_length, frame_field.point
    )
    # frame_field.save_active_frame_set(frame_path)
    # frame_field.load(frame_path)

    normal_scale = 0.1
    see_boundary = False
    draw_grid = False
    only_see_one_cell = False
    cell_3dindex_to_view = sdf_infill.grid.cell_3dcount // 2

    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    frame3_drawer = atom.drawer3.FrameFieldDrawer()
    frame3_drawer.init_from_field(frame_field)

    subdivisions = np.array([1024, 1024])
    grid_mesh2 = atom.drawer3.GridMesh2()
    grid_mesh2.create(subdivisions)
    grid_mesh2.origin = np.array(
        [
            bpn.bounding_box[1] * 0.5,
            bpn.bounding_box[3] * 0.5,
            bpn.bounding_box[5] * 0.5,
        ]
    )
    grid_mesh2.size = max(bpn.bounding_box) * 1.1
    grid_mesh2.orientation[0] = ti.math.pi * 0.5
    grid_mesh2.orientation[1] = ti.math.pi * 0.5

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 1.5, bpn.bounding_box[3] * 0.5
    )
    camera.lookat(
        bpn.bounding_box[1] * 0.5, bpn.bounding_box[5] * 0.5, -bpn.bounding_box[3] * 0.5
    )
    camera.up(0, 1, 0)
    # camera.projection_mode(ti.ui.ProjectionMode.Orthogonal)
    camera.fov(45 * 0.5)
    gui = window.get_gui()

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "b":
                if see_boundary:
                    see_boundary = False
                else:
                    see_boundary = True

            if window.event.key == "g":
                if draw_grid:
                    draw_grid = False
                else:
                    draw_grid = True
        with gui.sub_window("Parameters", 0.05, 0.05, 0.9, 0.26) as w:
            grid_mesh2.size = w.slider_float(
                "Grid mesh size",
                grid_mesh2.size,
                min(bpn.bounding_box),
                max(bpn.bounding_box),
            )
            grid_mesh2.orientation[0] = w.slider_float(
                "Theta", grid_mesh2.orientation[0], 0.0, ti.math.pi
            )
            grid_mesh2.orientation[1] = w.slider_float(
                "Phi", grid_mesh2.orientation[1], 0.0, ti.math.pi * 2.0
            )
            grid_mesh2.origin[0] = w.slider_float(
                "Origin X", grid_mesh2.origin[0], 0.0, bpn.bounding_box[1]
            )
            grid_mesh2.origin[1] = w.slider_float(
                "Origin Y", grid_mesh2.origin[1], 0.0, bpn.bounding_box[3]
            )
            grid_mesh2.origin[2] = w.slider_float(
                "Origin Z", grid_mesh2.origin[2], 0.0, bpn.bounding_box[5]
            )
            cell_3dindex_to_view[0] = w.slider_int(
                "Cell 3D index to view X",
                cell_3dindex_to_view[0],
                0,
                triphasor3_field.grid.cell_3dcount[0] - 1,
            )
            cell_3dindex_to_view[1] = w.slider_int(
                "Cell 3D index to view Y",
                cell_3dindex_to_view[1],
                0,
                triphasor3_field.grid.cell_3dcount[0] - 1,
            )
            cell_3dindex_to_view[2] = w.slider_int(
                "Cell 3D index to view Z",
                cell_3dindex_to_view[2],
                0,
                triphasor3_field.grid.cell_3dcount[0] - 1,
            )

        camera.track_user_inputs(window, movement_speed=0.2, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        grid_mesh2.update_vertex_normal()
        grid_mesh2.update_per_vertex_color_with_triphasor3_field_1ex(
            triphasor3_field, True, draw_grid
        )
        frame3_drawer.update(frame_field, cell_3dindex_to_view, only_see_one_cell)

        scene.mesh(
            grid_mesh2.vertex,
            grid_mesh2.index,
            grid_mesh2.normal,
            show_wireframe=False,
            per_vertex_color=grid_mesh2.per_vertex_color,
        )
        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )
        scene.particles(
            frame3_drawer.points,
            sdf_infill.grid.cell_sides_length * 0.1,
            color=(1.0, 0.0, 0.0),
        )
        scene.lines(
            frame3_drawer.lines,
            width=2,
            color=(1.0, 0.0, 0.0),
        )

        canvas.scene(scene)
        window.show()

    ti.reset()


if __name__ == "__main__":
    # To see how a tricosine is aligned with another one
    experiment_align_triangle_i_with_j()

    # To see how a tricosine field is globally aligned
    # experiment_triphasor3_extract_maxima_1ex()
