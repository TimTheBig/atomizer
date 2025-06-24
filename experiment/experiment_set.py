import taichi as ti

import atom.set

ti.init(arch=ti.cpu, debug=True)


def experiment_set():
    my_set = atom.set.Set()
    my_set.create_u32(10)

    print(my_set.value)
    print(my_set.size)
    my_set.insert_u32(12)
    print(my_set.value)
    print(my_set.size)
    my_set.insert_u32(13)
    my_set.insert_u32(14)
    print(my_set.value)
    print(my_set.size)
    my_set.remove_u32(12)
    print(my_set.value)
    print(my_set.size)


if __name__ == "__main__":
    experiment_set()
