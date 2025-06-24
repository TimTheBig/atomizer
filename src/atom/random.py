import taichi as ti

from . import limits, math


@ti.func
def pcg(v: ti.u32) -> ti.u32:
    # https://www.pcg-random.org/
    state = v * ti.u32(747796405) + ti.u32(2891336453)
    word = ((state >> ((state >> 28) + ti.u32(4))) ^ state) * ti.u32(277803737)
    return (word >> 22) ^ word


@ti.func
def pcgf(v: ti.u32) -> float:
    ui = pcg(v)
    uf = ti.cast(ui, float) / ti.cast(ti.u32(limits.u32_max), float)
    return uf


@ti.func
def pcg2d(v: ti.math.uvec2) -> ti.math.uvec2:
    """
    https://www.shadertoy.com/view/XlGcRh
    """
    v = v * ti.u32(1664525) + ti.u32(1013904223)

    v.x = v.x + v.y * ti.u32(1664525)
    v.y = v.y + v.x * ti.u32(1664525)

    v = v ^ (v >> ti.u32(16))

    v.x = v.x + v.y * ti.u32(1664525)
    v.y = v.y + v.x * ti.u32(1664525)

    v = v ^ (v >> ti.u32(16))

    return v


@ti.func
def pcg2df(v: ti.math.uvec2) -> ti.math.vec2:
    ui = pcg2d(v)
    uf = ti.cast(ui, float) / ti.cast(ti.u32(limits.u32_max), float)
    return uf


@ti.func
def pcg3d(v: ti.math.uvec3) -> ti.math.uvec3:
    """
    // http://www.jcgt.org/published/0009/03/02/
    """

    v = v * ti.u32(1664525) + ti.u32(1013904223)

    v.x += v.y * v.z
    v.y += v.z * v.x
    v.z += v.x * v.y

    v ^= v >> ti.u32(16)

    v.x += v.y * v.z
    v.y += v.z * v.x
    v.z += v.x * v.y

    return v


@ti.func
def pcg3df(v: ti.math.uvec3) -> ti.math.vec3:
    ui = pcg3d(v)
    uf = ti.cast(ui, float) / ti.cast(ti.u32(limits.u32_max), float)
    return uf


@ti.func
def pcg_5_to_1(v: math.uvec5) -> ti.u32:
    return pcg(v[0] + pcg(v[1] + pcg(v[2] + pcg(v[3] + pcg(v[4])))))


@ti.func
def pcg_7_to_1(v: math.uvec7) -> ti.u32:
    return pcg(
        v[0] + pcg(v[1] + pcg(v[2] + pcg(v[3] + pcg(v[4] + pcg(v[5] + pcg(v[6]))))))
    )


@ti.func
def pcgf_5_to_1(v: math.uvec5) -> float:
    ui = pcg_5_to_1(v)
    uf = ti.cast(ui, float) / ti.cast(ti.u32(limits.u32_max), float)
    return uf


@ti.func
def pcgf_7_to_1(v: math.uvec7) -> float:
    ui = pcg_7_to_1(v)
    uf = ti.cast(ui, float) / ti.cast(ti.u32(limits.u32_max), float)
    return uf
