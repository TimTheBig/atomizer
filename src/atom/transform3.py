import taichi as ti


@ti.func
def compute_frame_to_canonical_matrix(
    axis0: ti.math.vec3, axis1: ti.math.vec3, axis2: ti.math.vec3, origin: ti.math.vec3
) -> ti.math.mat3:
    """
    Compute the frame to canonical matrix that transforms points and vectors
    expressed in the frame space to the same points expressed in the canonical
    frame.

    Parameters
    ----------
    axis0, axis1, origin : ti.math.vec2
        The axis 0 and 1, and the origin, of the frame expressed in the canonical matrix.

    Notes
    -----
        Marschner and Shirley, Fundamentals of Computer Graphics, 5th edition, p. 153
    """
    return ti.math.mat4(
        [
            [axis0.x, axis1.x, axis2.x, origin.x],
            [axis0.y, axis1.y, axis2.y, origin.y],
            [axis0.z, axis1.z, axis2.z, origin.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def apply_to_point(T: ti.math.mat4, p: ti.math.vec3) -> ti.math.vec3:
    p_homogeneous = ti.math.vec4(p, 1.0)
    p_transformed = T @ p_homogeneous
    p_transformed /= p_transformed[3]
    return p_transformed[:3]


@ti.func
def apply_to_vector(T: ti.math.mat4, v: ti.math.vec3) -> ti.math.vec3:
    v_homogeneous = ti.math.vec4(v, 0.0)
    v_transformed = T @ v_homogeneous
    return v_transformed.xyz


@ti.func
def scale_uniformly(s: float) -> ti.math.mat4:
    return ti.math.mat4(
        [
            [s, 0.0, 0.0, 0.0],
            [0.0, s, 0.0, 0.0],
            [0.0, 0.0, s, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def translate(t: ti.math.vec3) -> ti.math.mat4:
    return ti.math.mat4(
        [
            [1.0, 0.0, 0.0, t.x],
            [0.0, 1.0, 0.0, t.y],
            [0.0, 0.0, 1.0, t.z],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def rotate_x(theta: float) -> ti.math.mat4:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return ti.math.mat4(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, cos_theta, -sin_theta, 0.0],
            [0.0, sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def rotate_y(theta: float) -> ti.math.mat4:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return ti.math.mat4(
        [
            [cos_theta, 0.0, sin_theta, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-sin_theta, 0.0, cos_theta, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


@ti.func
def rotate_z(theta: float) -> ti.math.mat4:
    cos_theta = ti.cos(theta)
    sin_theta = ti.sin(theta)
    return ti.math.mat4(
        [
            [cos_theta, -sin_theta, 0.0, 0.0],
            [sin_theta, cos_theta, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
