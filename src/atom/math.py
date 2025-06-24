import numpy as np
import taichi as ti

from . import limits

vec5 = ti.types.vector(5, float)
vec6 = ti.types.vector(6, float)
vec7 = ti.types.vector(7, float)
vec8 = ti.types.vector(8, float)
vec9 = ti.types.vector(9, float)
vec10 = ti.types.vector(10, float)
vec12 = ti.types.vector(12, float)
vec13 = ti.types.vector(13, float)
vec20 = ti.types.vector(20, float)

uvec5 = ti.types.vector(5, ti.u32)
uvec7 = ti.types.vector(7, ti.u32)

imat4x3 = ti.types.matrix(4, 3, int)
mat2x5 = ti.types.matrix(2, 5, float)
mat4x2 = ti.types.matrix(4, 2, float)
mat4x3 = ti.types.matrix(4, 3, float)

FLOAT_EPSILON = 1.2e-7


def roundup_power_of_2(x: int) -> int:
    log2_x = np.log2(x)
    log2_x_ceil = np.ceil(log2_x).astype(np.int_)
    return 2**log2_x_ceil


@ti.func
def atan(x):
    """
    atan is not available in the Taichi language.
    """
    atan_val = ti.math.atan2(x, 1.0)
    return atan_val


@ti.func
def acos_safe(x):
    x = ti.math.clamp(x, -1.0, 1.0)
    return ti.math.acos(x)


@ti.func
def normalize_safe(x):
    x_normalized = ti.math.normalize(x)
    x_nan = ti.math.isnan(x_normalized)
    if x_nan.any():
        x_normalized.fill(0.0)
        x_normalized[0] = 1.0
    return x_normalized


@ti.func
def eval_triangle_filter(x, mean, radius: float):
    """The function evaluates the triangle filter.

    Notes
    -----
    https://www.pbr-book.org/3ed-2018/Texture/Image_Texture#IsotropicTriangleFilter
    """
    x_mean = x - mean
    return ti.max(0.0, radius - ti.math.length(x_mean))


@ti.func
def eval_triangle_filter_normalized(x, mean, radius: float):
    return eval_triangle_filter(x, mean, radius) / radius


@ti.func
def eigenvec2_with_highest_eigenvalue_iterative(A: ti.math.mat2) -> ti.math.vec2:
    b_0 = ti.math.vec2(1.0, 0.0)
    if A[0, 0] < A[1, 1]:
        b_0 = ti.math.vec2(0.0, 1.0)

    for _ in range(16):
        b_0 = normalize_safe(A @ b_0)
    return b_0


@ti.kernel
def eval_min_max(scalars: ti.template()) -> ti.math.vec2:
    """
    Evaluate the minimum and maximum values of a scalar field.
    """
    min = limits.f32_max
    max = limits.f32_min
    for i in ti.grouped(scalars):
        s_i = scalars[i]
        if ti.math.isnan(s_i):
            continue
        if s_i == limits.f32_max:
            continue

        ti.atomic_max(max, s_i)
        ti.atomic_min(min, s_i)
    return ti.math.vec2(min, max)
