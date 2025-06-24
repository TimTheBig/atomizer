import sys
import time
from math import pi

import numpy as np
import taichi as ti
from tqdm import tqdm

import atom.color
import atom.drawer3
import atom.fff3
import atom.frame3
import atom.solid3
import atom.toolpath3
from atom.bvh import BVH


def experiment_region_drawer_one_frame():
    ti.init(arch=ti.gpu, debug=True)

    frame = atom.frame3.Set()
    frame_dict = {}
    frame_dict["point"] = np.array([[0.0, 0.0, 0.0]])
    frame_dict["normal"] = np.array([[0.0, 0.0]])
    frame_dict["phi_t"] = np.array([0.0])
    frame.from_numpy(frame_dict)

    layer_height = 0.4
    triperiod = np.array([layer_height, layer_height * 2.0, layer_height])

    subdivisions = np.array([1024, 1024])
    grid_mesh2 = atom.drawer3.GridMesh2()
    grid_mesh2.create(subdivisions)
    grid_mesh2.origin = np.array([0.0, 0.0, 0.0])
    grid_mesh2.size = 8.0 * layer_height
    grid_mesh2.orientation[0] = np.pi * 0.5
    grid_mesh2.orientation[1] = np.pi * 0.5

    frame_drawer = atom.drawer3.FrameSetDrawer()
    frame_drawer.init_from_field(frame)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(1.0, 1.0, 1.0)
    camera.lookat(0.0, 0.0, 0.0)
    camera.up(0, 1, 0)
    camera.fov(45 * 0.5)

    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            frame.normal[0][0] = w.slider_float(
                "theta_n", frame.normal[0][0], 0.0, ti.math.pi
            )
            frame.normal[0][1] = w.slider_float(
                "phi_n", frame.normal[0][1], -ti.math.pi, ti.math.pi
            )
            frame.phi_t[0] = w.slider_float(
                "phi_t", frame.phi_t[0], -ti.math.pi, ti.math.pi
            )
            grid_mesh2.orientation[0] = w.slider_float(
                "Theta", grid_mesh2.orientation[0], 0.0, ti.math.pi
            )
            grid_mesh2.orientation[1] = w.slider_float(
                "Phi", grid_mesh2.orientation[1], 0.0, ti.math.pi * 2.0
            )
            grid_mesh2.origin[0] = w.slider_float(
                "Origin X", grid_mesh2.origin[0], -3.0, 3.0
            )
            grid_mesh2.origin[1] = w.slider_float(
                "Origin Y", grid_mesh2.origin[1], -3.0, 3.0
            )
            grid_mesh2.origin[2] = w.slider_float(
                "Origin Z", grid_mesh2.origin[2], -3.0, 3.0
            )
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        grid_mesh2.update_vertex_normal()
        grid_mesh2.update_per_vertex_color_with_frame_set_region(frame, triperiod, 0)

        frame_drawer.update(frame, np.array([0, frame.phi_t.shape[0]]), layer_height)

        scene.particles(
            frame_drawer.point,
            0.01,
            color=tuple(atom.color.class3dark20),
        )
        scene.lines(
            frame_drawer.line_vertex,
            width=2,
            per_vertex_color=frame_drawer.line_per_vertex_color,
        )
        scene.mesh(
            grid_mesh2.vertex,
            grid_mesh2.index,
            grid_mesh2.normal,
            show_wireframe=False,
            per_vertex_color=grid_mesh2.per_vertex_color,
        )

        canvas.scene(scene)
        window.show()


def experiment_region_drawer_frame_set():
    ti.init(arch=ti.gpu, debug=False)

    solid_name = "triangle_24"
    frame_path = f"data/frame/{solid_name}.npz"

    frame_set = atom.frame3.Set()
    frame_set.load(frame_path)
    p_min, p_max = frame_set.get_aabb()
    side_length = p_max - p_min

    layer_height = 0.4
    triperiod = np.array([layer_height, layer_height * 2.0, layer_height])
    atom_index = 159
    atom_min = 0
    atom_max = frame_set.phi_t.shape[0] - 1

    subdivisions = np.array([1024, 1024])
    grid_mesh2 = atom.drawer3.GridMesh2()
    grid_mesh2.create(subdivisions)
    grid_mesh2.origin = np.array(
        [side_length[0] * 0.5, side_length[1] * 0.5, side_length[2] * 0.5]
    )
    grid_mesh2.size = max(side_length)
    grid_mesh2.orientation[0] = np.pi * 0.5
    grid_mesh2.orientation[1] = np.pi * 0.5

    frame_drawer = atom.drawer3.FrameSetDrawer()
    frame_drawer.init_from_field(frame_set)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(side_length[0] * 0.5, side_length[2] * 1.5, side_length[1] * 0.5)
    camera.lookat(side_length[0] * 0.5, side_length[2] * 0.5, -side_length[1] * 0.5)
    camera.up(0, 1, 0)
    camera.fov(45 * 0.5)

    gui = window.get_gui()

    while window.running:
        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            grid_mesh2.orientation[0] = w.slider_float(
                "Theta", grid_mesh2.orientation[0], 0.0, ti.math.pi
            )
            grid_mesh2.orientation[1] = w.slider_float(
                "Phi", grid_mesh2.orientation[1], 0.0, ti.math.pi * 2.0
            )
            grid_mesh2.origin[0] = w.slider_float(
                "Origin X", grid_mesh2.origin[0], -3.0, 3.0
            )
            grid_mesh2.origin[1] = w.slider_float(
                "Origin Y", grid_mesh2.origin[1], -3.0, 3.0
            )
            grid_mesh2.origin[2] = w.slider_float(
                "Origin Z", grid_mesh2.origin[2], -3.0, 3.0
            )
            atom_index = w.slider_int(
                "Atom index", atom_index, 0, frame_set.phi_t.shape[0] - 1
            )
            atom_min = w.slider_int(
                "Atom min", atom_min, 0, frame_set.phi_t.shape[0] - 1
            )
            atom_max = w.slider_int(
                "Atom max", atom_max, 0, frame_set.phi_t.shape[0] - 1
            )
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((1.0, 1.0, 1.0))

        grid_mesh2.update_vertex_normal()
        grid_mesh2.update_per_vertex_color_with_frame_set_region(
            frame_set, triperiod, atom_index
        )

        frame_drawer.update(frame_set, np.array([atom_min, atom_max]), layer_height)

        scene.particles(
            frame_drawer.point,
            0.01,
            color=tuple(atom.color.class3dark20),
        )
        scene.lines(
            frame_drawer.line_vertex,
            width=2,
            per_vertex_color=frame_drawer.line_per_vertex_color,
        )
        scene.mesh(
            grid_mesh2.vertex,
            grid_mesh2.index,
            grid_mesh2.normal,
            show_wireframe=False,
            per_vertex_color=grid_mesh2.per_vertex_color,
        )

        canvas.scene(scene)
        window.show()


def debug_atom_ordering():
    ti.init(arch=ti.cpu, debug=False, offline_cache_cleaning_policy="never")

    solid_name = "triangle_24"
    bpn_path = f"data/point_normal/{solid_name}.npz"
    sdf_path = f"data/sdf/{solid_name}.npz"
    frame_path = f"data/frame/{solid_name}.npz"

    bpn = atom.solid3.BoundaryPointNormal()
    bpn.load(bpn_path)

    sdf = atom.solid3.SDF()
    sdf.load(sdf_path)

    frame_set = atom.frame3.Set()
    frame_set.load(frame_path)
    p_min, p_max = frame_set.get_aabb()
    p_heighest = frame_set.get_highest_point()
    side_length = p_max - p_min
    domain_diag = np.linalg.norm(side_length)
    atom_count = frame_set.phi_t.shape[0]
    print(f"Atom count: {atom_count}")

    # Create BVH
    bvh = BVH(auto_rebuild=True)
    bvh.from_frame_set(frame_set)

    triperiod = atom.fff3.triperiod_from_cell_sides_length_kernel(
        sdf.grid.cell_sides_length
    ).to_numpy()
    atom_min = 0
    atom_max = frame_set.phi_t.shape[0] - 1

    neighborhood_radius = atom.fff3.neighborhood_radius_kernel(
        sdf.grid.cell_sides_length
    )
    print(f"neighborhood_radius: {neighborhood_radius}")
    toolpath_planner = atom.toolpath3.ToolpathPlanner()
    toolpath_planner.init(
        frame_set, triperiod, neighborhood_radius, domain_diag, bvh, p_heighest
    )

    cone_mesh = atom.drawer3.ConeMesh()
    cone_mesh.create(
        64,
        atom.toolpath3.NOZZLE_CONE_ANGLE,
        4.0,
        toolpath_planner.p_0,
        np.array([0.0, 0.0], dtype=np.float32),
    )

    any_bidi_constraints = toolpath_planner.check_for_bidirectional_constraints()
    if any_bidi_constraints:
        sys.exit(
            "The frame set has one or several bidirectional constraints, making the ordering impossible. Please regenerate the atoms with softer constraints on the normals."
        )

    frame_drawer = atom.drawer3.FrameSetDrawer()
    frame_drawer.init_from_field(toolpath_planner.frame_set)

    frame_data_drawer = atom.drawer3.FrameDataSetDrawer()
    frame_data_drawer.allocate(toolpath_planner.frame_data_set)

    normal_scale = 0.1
    see_boundary = False
    bpn_drawer = atom.drawer3.BoundaryPointNormalDrawer()
    bpn_drawer.init_from_bpn(bpn, normal_scale)

    window = ti.ui.Window(
        name="Window Title", res=(1280, 720), fps_limit=200, pos=(150, 150), vsync=True
    )
    canvas = window.get_canvas()
    scene = window.get_scene()

    camera = ti.ui.Camera()
    camera.position(side_length[0] * 0.5, side_length[2] * 1.5, side_length[1] * 0.5)
    camera.lookat(side_length[0] * 0.5, side_length[2] * 0.5, -side_length[1] * 0.5)
    camera.up(0, 1, 0)
    camera.fov(45 * 0.5)

    gui = window.get_gui()

    advance = True
    always_advancing = False
    update_state = True
    find_best_next = False

    while window.running:
        if window.get_event(ti.ui.PRESS):
            if window.event.key == "n":
                advance = True
            if window.event.key == "x":
                if always_advancing:
                    always_advancing = False
                    advance = False
                else:
                    always_advancing = True
                    advance = True
            if window.event.key == "b":
                if see_boundary:
                    see_boundary = False
                else:
                    see_boundary = True
            if window.event.key == "v":
                current_filename = f"data/toolpath_planner/{solid_name}_{toolpath_planner.iteration_number:06d}.npz"
                toolpath_planner.save(current_filename)
            if window.event.key == "c":
                current_filename = f"data/toolpath/{solid_name}_{toolpath_planner.iteration_number:06d}.npz"
                toolpath_np = atom.toolpath3.Toolpath()
                toolpath_np.from_numpy(toolpath_planner.toolpath.to_numpy())
                toolpath_planner.toolpath = toolpath_np
                # Uplift by half a layer
                toolpath_planner.toolpath.uplift(triperiod[2] * 0.5)
                toolpath_planner.toolpath.flip()
                toolpath_planner.toolpath.save(current_filename)

        with gui.sub_window("Parameters", 0.05, 0.05, 0.41, 0.15) as w:
            atom_min = w.slider_int(
                "Atom min",
                atom_min,
                0,
                toolpath_planner.frame_set.phi_t.shape[0] - 1,
            )
            atom_max = w.slider_int(
                "Atom max",
                atom_max,
                0,
                toolpath_planner.frame_set.phi_t.shape[0] - 1,
            )
        camera.track_user_inputs(window, movement_speed=0.1, hold_key=ti.ui.RMB)
        scene.set_camera(camera)
        scene.ambient_light((0.5, 0.5, 0.5))
        scene.point_light(
            pos=(domain_diag, domain_diag, domain_diag), color=(1.0, 1.0, 1.0)
        )

        if advance and toolpath_planner.iteration_number != atom_count:
            if update_state:
                print("Update state")
                update_state_t0 = time.perf_counter()

                if toolpath_planner.iteration_number == 0:
                    toolpath_planner.init_non_supporting_set()
                else:
                    # DEBUG
                    # toolpath_planner.init_non_supporting_set()
                    if not toolpath_planner.backpropagate:
                        toolpath_planner.update_non_supporting_set()

                # print(
                #     f"Non supporting set size: {toolpath_planner.non_supporting_frame_set.size}"
                # )
                # print(
                #     f"Non supporting set size max: {toolpath_planner.non_supporting_frames_set_size_max}"
                # )

                # toolpath_planner.init_non_supporting_state()
                # toolpath_planner.init_non_supporting_state_brute_force()
                # toolpath_planner.update_relative_state_and_distance()
                # toolpath_planner.init_unaccessibility()

                # DEBUG
                # toolpath_planner.init_non_supporting_set()

                toolpath_planner.init_non_supporting_state_fast_track_option()

                # DEBUG
                # toolpath_planner.init_unaccessibility_brute_force()

                if not toolpath_planner.fast_track:
                    toolpath_planner.init_atom_in_a_cycle()

                toolpath_planner.compute_cost()

                update_state_t1 = time.perf_counter()

                print(
                    f"Update state {toolpath_planner.iteration_number} took {update_state_t1 - update_state_t0} s. backpropagate was {toolpath_planner.backpropagate}. fast_track was {toolpath_planner.fast_track}. non supporting set size is {toolpath_planner.non_supporting_frame_set.size}."
                )
            if find_best_next:
                next_index, next_travel_type = toolpath_planner.find_best_next()
                print(f"next_travel_type: {next_travel_type}")
                print(f"next_index: {next_index}")

                if next_index == -1 and toolpath_planner.backpropagate:
                    if toolpath_planner.iteration_number == atom_count:
                        current_filename = f"data/toolpath/{solid_name}.npz"
                        toolpath_np = atom.toolpath3.Toolpath()
                        toolpath_np.from_numpy(toolpath_planner.toolpath.to_numpy())
                        toolpath_planner.toolpath = toolpath_np
                        toolpath_planner.toolpath.flip()
                        # Uplift by half a layer
                        toolpath_planner.toolpath.uplift(triperiod[2] * 0.5)
                        toolpath_planner.toolpath.save(current_filename)
                        # To avoid saving and flipping again the toolpath
                        toolpath_planner.iteration_number += 1
                    else:
                        sys.exit(
                            f"No more valid next atom, even with the backpropagate, but there is still atoms in the set. Maybe the set of non supporting atoms is corrupted and you should use toolpath_planner.init_non_supporting_set() to reinitialize it. Current iteration number: {toolpath_planner.iteration_number}. Or maybe two adjacent atoms are constraining each other."
                        )

                if next_index != -1:
                    # next_is_adjacent_to_constraint_index = (
                    #     toolpath_planner.atom_i_is_adjacent_to_constraint(next_index)
                    # )
                    # next_is_adjacent_to_constraint = (
                    #     next_is_adjacent_to_constraint_index != -1
                    # )
                    # if next_is_adjacent_to_constraint:
                    #     print(
                    #         f"next_is_adjacent_to_constraint: {next_is_adjacent_to_constraint_index}"
                    #     )

                    next_is_depo = (
                        next_travel_type == atom.toolpath3.TRAVEL_TYPE_DEPOSITION
                    )

                    next_is_not_depo = not next_is_depo
                    current_is_not_depo = not (
                        toolpath_planner.current_travel_type
                        == atom.toolpath3.TRAVEL_TYPE_DEPOSITION
                    )
                    is_one_atom_length_trajectory = (
                        next_is_not_depo and current_is_not_depo
                    )

                    # backpropagate; Next is depo;
                    # 00: fast_track: false; update_toolpath: false; backpropagate: true

                    # 01 or
                    # 10 or
                    # 11: fast_track: true; update_toolpath: true; backpropagate: false

                    # backpropagate_required = (not (toolpath_planner.backpropagate or next_is_depo or not next_is_adjacent_to_constraint) and not toolpath_planner.iteration_number == 0)
                    backpropagate_required = (
                        not toolpath_planner.backpropagate
                        and (not next_is_depo or next_index == -1)
                        and not toolpath_planner.iteration_number == 0
                    )

                    # DEBUG
                    # backpropagate_required = False

                    if backpropagate_required:
                        toolpath_planner.fast_track = False
                        toolpath_planner.backpropagate = True
                        print("backpropagate_required")
                    else:
                        # next_is_adjacent_to_constraint_index = (
                        #     toolpath_planner.atom_i_is_adjacent_to_constraint(
                        #         next_index
                        #     )
                        # )
                        # next_is_adjacent_to_constraint = (
                        #     next_is_adjacent_to_constraint_index != -1
                        # )
                        # if next_is_adjacent_to_constraint:
                        #     print(
                        #         f"next_is_adjacent_to_constraint: {next_is_adjacent_to_constraint_index}"
                        #     )
                        if next_is_not_depo:
                            next_is_reachable_by_constraint_index = 0
                            next_list = []
                            while next_is_reachable_by_constraint_index != -1:
                                next_list.append(next_index)
                                next_is_reachable_by_constraint_index = toolpath_planner.is_atom_i_reachable_by_a_constraint(
                                    next_index
                                )
                                if next_is_reachable_by_constraint_index != -1:
                                    print(
                                        f"next_is_reachable_by_constraint: {next_is_reachable_by_constraint_index}"
                                    )

                                    next_index, next_travel_type = (
                                        toolpath_planner.find_best_next_constrainer(
                                            next_is_reachable_by_constraint_index
                                        )
                                    )
                                    # If there is a circular constraint
                                    if next_index in next_list:
                                        break

                        toolpath_planner.fast_track = True
                        # DEBUG
                        # toolpath_planner.fast_track = False
                        if (
                            not is_one_atom_length_trajectory
                            or toolpath_planner.toolpath_point_count_previous == 0
                        ):
                            toolpath_planner.update_toolpath(
                                next_index, next_travel_type
                            )
                        else:
                            print("is_one_atom_length_trajectory")
                            # Get back to the previous point
                            toolpath_planner.toolpath.point_count = (
                                toolpath_planner.toolpath_point_count_previous
                            )
                            toolpath_planner.current_frame_index = (
                                toolpath_planner.previous_frame_index
                            )
                            toolpath_planner.current_point = (
                                toolpath_planner.previous_point
                            )
                            toolpath_planner.update_toolpath(
                                next_index, next_travel_type
                            )
                        toolpath_planner.backpropagate = False
                else:
                    # For an unknow reason, the set of non supporting can be unexact,
                    # most probably because the update_non_supporing_set can miss one atom.
                    toolpath_planner.init_non_supporting_set()
                    toolpath_planner.fast_track = False
                    toolpath_planner.backpropagate = True
                    print("backpropagate_required, no more unconstrained atoms")

                cone_mesh.origin = toolpath_planner.current_point
                cone_mesh.orientation = toolpath_planner.current_normal_sph

            if not always_advancing:
                advance = False
            if update_state:
                update_state = False
                find_best_next = True
            elif find_best_next:
                update_state = True
                find_best_next = False

        frame_drawer.update(
            toolpath_planner.frame_set, np.array([atom_min, atom_max]), triperiod[2]
        )
        cone_mesh.update_vertex_normal()
        # toolpath_drawer.update(
        #     toolpath_planner.toolpath, toolpath_planner.toolpath.point_count
        # )

        # frame_data_drawer.draw_non_supporting(toolpath_planner.frame_data_set)
        # frame_data_drawer.draw_unaccessible(toolpath_planner.frame_data_set)
        # frame_data_drawer.draw_atoms_in_a_cycle(toolpath_planner.frame_data_set)
        # frame_data_drawer.draw_atoms_any_below(toolpath_planner.frame_data_set)
        # frame_data_drawer.draw_wall(toolpath_planner.frame_data_set)
        frame_data_drawer.draw_cost(toolpath_planner.frame_data_set)

        scene.mesh(
            cone_mesh.vertex,
            cone_mesh.index,
            cone_mesh.normal,
            show_wireframe=False,
            color=(0.5, 0.5, 0.5),
        )
        scene.particles(
            frame_drawer.point,
            0.02,
            per_vertex_color=frame_data_drawer.point_color,
        )
        scene.lines(
            frame_drawer.line_vertex,
            width=2,
            per_vertex_color=frame_drawer.line_per_vertex_color,
        )
        if see_boundary:
            scene.lines(
                bpn_drawer.line_vertex,
                width=2,
                per_vertex_color=bpn_drawer.per_vertex_color,
            )

        # scene.particles(
        #     toolpath_drawer.vertex,
        #     0.02,
        #     color=atom.color.class3set21,
        # )
        # scene.lines(
        #     toolpath_drawer.line_vertex,
        #     width=2,
        #     per_vertex_color=toolpath_drawer.per_vertex_color,
        # )

        canvas.scene(scene)
        window.show()

    ti.reset()


if __name__ == "__main__":
    experiment_region_drawer_one_frame()
    # experiment_region_drawer_frame_set()
    # debug_atom_ordering()
