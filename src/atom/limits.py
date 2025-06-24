"""
Limits of primitive types

https://www.h-schmidt.net/FloatConverter/
"""

import numpy as np

f32_max = np.finfo(np.float32).max
f32_min = np.finfo(np.float32).min

i32_max = 2147483647

u32_max = 4294967295
