import taichi as ti

import atom.basis3
import atom.color
import atom.direction
import atom.drawer3
import atom.fff3
import atom.line
import atom.solid3
import atom.transform3

ti.init(arch=ti.gpu)


@ti.kernel
def init_vertices_per_vertex_color(
    vertices: ti.template(), per_vertex_color: ti.template()
):
    vertices[0] = ti.math.vec3(1.0, 0.0, 0.0)
    vertices[1] = ti.math.vec3(-1.0, 0.0, 0.0)
    vertices[2] = ti.math.vec3(0.0, 1.0, 0.0)
    vertices[3] = ti.math.vec3(0.0, -1.0, 0.0)
    vertices[4] = ti.math.vec3(0.0, 0.0, 1.0)
    vertices[5] = ti.math.vec3(0.0, 0.0, -1.0)
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
def from_spherical_basis_kernel(
    spherical_basis: ti.math.vec3, line_vertices: ti.template()
):
    basis = atom.basis3.from_spherical(spherical_basis)
    t = ti.math.vec3(basis[0, 0], basis[1, 0], basis[2, 0])
    b = ti.math.vec3(basis[0, 1], basis[1, 1], basis[2, 1])
    n = ti.math.vec3(basis[0, 2], basis[1, 2], basis[2, 2])

    frame_to_canonical = atom.transform3.compute_frame_to_canonical_matrix(
        t, b, n, ti.math.vec3(0.0, 0.0, 0.0)
    )

    for i in range(line_vertices.shape[0] // 2):
        start_point_i = line_vertices[i * 2]
        end_point_i = line_vertices[i * 2 + 1]
        line_vertices[i * 2] = atom.transform3.apply_to_point(
            frame_to_canonical, start_point_i
        )
        line_vertices[i * 2 + 1] = atom.transform3.apply_to_point(
            frame_to_canonical, end_point_i
        )


def experiment_from_spherical_basis():
    theta_n = 0.0
    phi_n = 0.0
    phi_t = 0.0

    basis_vertices = ti.Vector.field(3, dtype=ti.f32, shape=6)
    basis_per_vertex_color = ti.Vector.field(3, dtype=ti.f32, shape=6)

    spherical_basis = ti.math.vec3(theta_n, phi_n, phi_t)

    window = ti.ui.Window("Basis", (768, 768))
    canvas = window.get_canvas()
    gui = window.get_gui()
    scene = window.get_scene()
    camera = ti.ui.Camera()
    camera.position(0.0, 0.0, 2.0)

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            theta_n = w.slider_float("theta_n", theta_n, 0.0, ti.math.pi)
            phi_n = w.slider_float("phi_n", phi_n, -ti.math.pi, ti.math.pi)
            phi_t = w.slider_float("phi_t", phi_t, -ti.math.pi, ti.math.pi)

        spherical_basis = ti.math.vec3(theta_n, phi_n, phi_t)

        init_vertices_per_vertex_color(basis_vertices, basis_per_vertex_color)
        from_spherical_basis_kernel(spherical_basis, basis_vertices)

        camera.track_user_inputs(window, movement_speed=0.04, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.8, 0.8, 0.8))
        scene.point_light(pos=(0.5, 1.5, 1.5), color=(1, 1, 1))
        scene.lines(basis_vertices, width=5, per_vertex_color=basis_per_vertex_color)

        canvas.scene(scene)
        window.show()


if __name__ == "__main__":
    experiment_from_spherical_basis()
