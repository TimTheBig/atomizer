import bpy
import sys
import mathutils


def import_stl(filepath):
    bpy.ops.wm.stl_import(filepath=filepath)
    obj = bpy.context.selected_objects[0]  # Assume the imported object is selected
    return obj


def main():
    # Parse command line arguments
    args = sys.argv
    # Blender arguments end with '--', user arguments start after it
    if "--" in args:
        idx = args.index("--") + 1
        user_args = args[idx:]
    else:
        print("Error: No arguments passed after '--'")
        return

    if len(user_args) < 2:
        print(
            "Usage: blender -b -P process_for_atomizer.py -- <input_stl_path> <output_obj_path> <remesh_voxel_size>"
        )
        return

    input_stl_path = user_args[0]
    output_obj_path = user_args[1]
    remesh_voxel_size = float(user_args[2])

    # Import STL
    imported_obj = import_stl(input_stl_path)
    if imported_obj.data.users > 1:
        imported_obj.data = imported_obj.data.copy()
    bpy.ops.object.shade_smooth()
    num_vertices = len(imported_obj.data.vertices)
    print(f"The selected object '{imported_obj.name}' has {num_vertices} vertices.")

    bpy.ops.object.shade_smooth()

    # translate the selected object such that its bounding box's minimal point aligns with (0, 0, 0)
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    bbox_corners = [
        imported_obj.matrix_world @ mathutils.Vector(corner)
        for corner in imported_obj.bound_box
    ]
    min_point = mathutils.Vector(
        (
            min(corner.x for corner in bbox_corners),
            min(corner.y for corner in bbox_corners),
            min(corner.z for corner in bbox_corners),
        )
    )
    # Compute the translation vector to align min_point to (0, 0, 0)
    translation_vector = -min_point

    # Apply the translation
    imported_obj.location += translation_vector

    # Remesh
    bpy.context.object.data.use_remesh_preserve_volume = False

    print(f"Remesh voxel size: {remesh_voxel_size} mm")
    bpy.context.object.data.remesh_voxel_size = remesh_voxel_size
    bpy.ops.object.voxel_remesh()

    # Add Smooth Modifier
    smooth_modifier = imported_obj.modifiers.new(name="Smooth", type="SMOOTH")

    # Set parameters
    smooth_modifier.factor = 0.9  # Adjust smooth strength (0 to 1)
    smooth_modifier.iterations = 10  # Number of smoothing iterations

    # Optional: Apply the modifier to make it permanent
    bpy.ops.object.modifier_apply(modifier="Smooth")

    bpy.ops.wm.obj_export(
        filepath=output_obj_path,
        export_selected_objects=True,
        export_triangulated_mesh=True,
        export_uv=False,
        export_normals=True,
        export_materials=False,
        up_axis="Z",
        forward_axis="Y",
    )


if __name__ == "__main__":
    main()
