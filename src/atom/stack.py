import taichi as ti
import taichi.math as tm


@ti.data_oriented
class Stack:
    def __init__(self, dtype, max_stack) -> None:
        self._field = ti.field(dtype=dtype, shape=(max_stack, ))
        self.index = ti.field(dtype=ti.int32, shape=())
        self.index[None] = -1
        
    @ti.func
    def push(self, val):
        self.index[None] += 1
        self._field[self.index[None]] = val
        
    @ti.func
    def pop(self):
        val = self._field[self.index[None]]
        self.index[None] -= 1
        return val
    
    @ti.func
    def clear(self):
        self.index[None] = -1
        
    @ti.func
    def length(self):
        return self.index[None] + 1
    
    @ti.func
    def is_empty(self):
        return self.index[None] < 0
    
    def to_list(self):
        ret = []
        for i in range(self.index[None] + 1):
            ret.append(self._field[i])
        return ret

