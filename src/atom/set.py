import numpy as np
import taichi as ti


class Set:
    def __init__(self) -> None:
        self.value = None
        self.size: int = None
        self.SIZE_MAX: int = None

    def to_numpy(self):
        dict_array = {}
        dict_array["value"] = self.value.to_numpy()
        dict_array["size"] = np.array(self.size)
        dict_array["size_max"] = np.array(self.SIZE_MAX)

        return dict_array

    def from_numpy(self, dict_array):
        self.SIZE_MAX = int(dict_array["size_max"][()])
        self.size = int(dict_array["size"][()])
        self.value = ti.field(dtype=ti.u32, shape=self.SIZE_MAX)

        self.value.from_numpy(dict_array["value"])

    def save(self, filename: str):
        dict_array = self.to_numpy()
        np.savez(filename, **dict_array)

    def load(self, filename: str):
        dict_array = np.load(filename)
        self.from_numpy(dict_array)

    def create_u32(self, SIZE_MAX: int):
        self.value = ti.field(dtype=ti.u32, shape=SIZE_MAX)
        self.size = 0
        self.SIZE_MAX = SIZE_MAX

    def insert_u32(self, element: int):
        if self.size < self.SIZE_MAX:
            set_insert_u32_kernel(self.value, self.size, element)
            self.size += 1
        else:
            exit("Set:insert_u32: Size max achieved. No more slots")

    def remove_u32(self, element: int):
        set_remove_u32_kernel(self.value, self.size, element)
        if self.size > 0:
            self.size -= 1


@ti.func
def set_insert_u32(field: ti.template(), size: int, element: ti.u32):
    field[size] = element


@ti.kernel
def set_insert_u32_kernel(field: ti.template(), size: int, element: ti.u32):
    set_insert_u32(field, size, element)


@ti.func
def set_remove_u32(field: ti.template(), size: int, element: ti.u32):
    found = 0

    ti.loop_config(serialize=True)
    for i in range(size):
        if field[i] == element:
            found = 1
        if found and i + 1 < size:
            field[i] = field[i + 1]


@ti.kernel
def set_remove_u32_kernel(field: ti.template(), size: int, element: ti.u32):
    set_remove_u32(field, size, element)


@ti.func
def set_find_u32(field: ti.template(), size: int, element: ti.u32) -> int:
    found = 0

    ti.loop_config(serialize=True)
    for i in range(size):
        if field[i] == element:
            found = 1
            break
    return found
