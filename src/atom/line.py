import numpy as np
import taichi as ti
from PIL import Image as PILImage


class Line2Field2:
    def __init__(self):
        self.angle = None

    def create_from_png_1channel_8bits(self, filename: str):
        im = PILImage.open(filename)
        pix = np.array(im).astype(np.float32) / 255.0
        pix = pix * np.pi - np.pi * 0.5
        pix = np.flip(pix, axis=0)
        pix = np.swapaxes(pix, 0, 1)
        self.angle = ti.field(dtype=ti.f32, shape=pix.shape)
        self.angle.from_numpy(pix)

    def create_from_normal_map(self, filename: str):
        im = PILImage.open(filename)
        pix = np.array(im).astype(np.float32) / 255.0
        pix = pix * 2.0 - 1.0
        pix_grayscale = np.arctan2(pix[:, :, 1], pix[:, :, 0])
        pix_grayscale = np.where(
            pix_grayscale < -np.pi * 0.5, pix_grayscale + np.pi, pix_grayscale
        )
        pix_grayscale = np.where(
            pix_grayscale > np.pi * 0.5, pix_grayscale - np.pi, pix_grayscale
        )

        pix_grayscale = np.flip(pix_grayscale, axis=0)
        pix_grayscale = np.swapaxes(pix_grayscale, 0, 1)
        self.angle = ti.field(dtype=ti.f32, shape=pix_grayscale.shape)
        self.angle.from_numpy(pix_grayscale)

    def save_to_png(self, filename):
        angle_np = self.angle.to_numpy()
        angle_normalized = (angle_np + np.pi * 0.5) / np.pi
        pix_grayscale_uint8 = (angle_normalized * 255).astype(np.uint8)
        pix_grayscale_uint8 = np.swapaxes(pix_grayscale_uint8, 0, 1)
        pix_grayscale_uint8 = np.flip(pix_grayscale_uint8, axis=0)
        im_out = PILImage.fromarray(pix_grayscale_uint8, mode="L")
        im_out.save(filename)


@ti.func
def line_field2_eval_nearest(s: ti.math.vec2, angle: ti.template()) -> float:
    """Evaluate the angle field using the nearest angle.

    Parameters
    ----------
    s : ndarray
        Continuous coordinates in $[0, 1]^2$.
    angle : field
        A 2D scalar field.

    Returns
    -------
    float
        The nearest angle from the continuous coordinates.
    """
    shape_xy = ti.math.ivec2(angle.shape)
    s_continuous = s * shape_xy
    s_discrete = ti.cast(ti.floor(s_continuous), int)
    s_discrete = ti.math.clamp(s_discrete, ti.math.ivec2(0, 0), shape_xy - 1)
    return angle[s_discrete]


@ti.func
def alignment_measure(l0, l1) -> float:
    return ti.math.dot(l0, l1) ** 2


@ti.func
def difference_measure(l0, l1) -> float:
    return (-1.0 * alignment_measure(l0, l1)) + 1.0
