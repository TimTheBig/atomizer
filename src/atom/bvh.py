import taichi as ti
import taichi.math as tm

MAX_PRIM_PER_NODE = 2
FLOAT_EPSILON = 1.2e-7
STACK_SIZE_MAX = 256
ivec32 = ti.types.vector(32, ti.i32)
ivecn = ti.types.vector(STACK_SIZE_MAX, ti.i32)


@ti.data_oriented
class BVH:
    def __init__(self, auto_rebuild=False, min_occupancy=0.5) -> None:
        self.npts = 0
        self.num_removed = ti.field(dtype=ti.i32, shape=())
        self.num_removed_since_rebuild = ti.field(dtype=ti.i32, shape=())

        self.node_field = None
        self.points = None
        self.point_id_field = None
        self.is_point_removed = None
        self.auto_rebuild = auto_rebuild
        self.min_occupancy = min_occupancy

    def from_numpy(self, point_coords):
        self.npts = len(point_coords)
        self.point_id_field = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.node_field = BVHNode.field(shape=(self.npts * 2 - 1,))

        # Convert from numpy arrays to taichi fields
        self.points = ti.Vector.field(n=3, dtype=ti.float32, shape=(self.npts,))
        self.points.from_numpy(point_coords)

        # Initialize flags for removed points
        self.is_point_removed = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.num_removed[None] = 0

        # Build hierarchy
        self.build()

    def from_bpn(self, bpn):
        self.npts = bpn.point.shape[0]
        self.point_id_field = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.node_field = BVHNode.field(shape=(self.npts * 2 - 1,))

        # Create local copy of point field
        self.points = ti.Vector.field(n=3, dtype=ti.float32, shape=(self.npts,))
        kernel_copy(from_field=bpn.point, to_field=self.points)

        # Initialize flags for removed points
        self.is_point_removed = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.num_removed[None] = 0

        # Build hierarchy
        self.build()

    def from_point_field(self, point_field):
        self.npts = point_field.shape[0]
        self.point_id_field = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.node_field = BVHNode.field(shape=(self.npts * 2 - 1,))

        # Create local copy of point field
        self.points = ti.Vector.field(n=3, dtype=ti.float32, shape=(self.npts,))
        kernel_copy(from_field=point_field, to_field=self.points)

        # Initialize flags for removed points
        self.is_point_removed = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.num_removed[None] = 0

        # Build hierarchy
        self.build()

    def from_frame_set(self, set):
        self.npts = set.point.shape[0]
        self.point_id_field = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.node_field = BVHNode.field(shape=(self.npts * 2 - 1,))

        # Create local copy of point fields
        self.points = ti.Vector.field(n=3, dtype=ti.float32, shape=(self.npts,))
        kernel_copy(from_field=set.point, to_field=self.points)

        # Initialize flags for removed points
        self.is_point_removed = ti.field(dtype=ti.int32, shape=(self.npts,))
        self.num_removed[None] = 0

        # Build hierarchy
        self.build()

    def build(self):
        # Call taich-scope bvh constructor
        self.num_removed_since_rebuild[None] = 0
        self._build_kernel()

    def rebuild(self):
        print("rebuild triggered")
        self.build()

    def remove_point(self, idx: ti.i32):
        prev_state = self.is_point_removed[idx]
        self.is_point_removed[idx] = 1
        if self.is_point_removed[idx] - prev_state == 1:
            self.num_removed[None] += 1
            self.num_removed_since_rebuild[None] += 1
            self.points[idx] = tm.vec3(
                10000, 10000, 10000
            )  # TODO: Replace with max f32
        if (1 - self.num_removed_since_rebuild[None] / self.npts) < self.min_occupancy:
            self.rebuild()

    @ti.func
    def build_ti_func(self):
        # Call taich-scope bvh constructor
        self.num_removed_since_rebuild[None] = 0
        self._build_ti_func()

    @ti.func
    def remove_point_ti_func(self, idx: int):
        prev_state = self.is_point_removed[idx]
        self.is_point_removed[idx] = 1
        if self.is_point_removed[idx] - prev_state == 1:
            self.num_removed[None] += 1
            self.num_removed_since_rebuild[None] += 1
            self.points[idx] = tm.vec3(
                10000, 10000, 10000
            )  # TODO: Replace with max f32
        if (1 - self.num_removed_since_rebuild[None] / self.npts) < self.min_occupancy:
            self.build_ti_func()

    @ti.func
    def _build_ti_func(self):
        # Initialize point id field
        for i in range(self.npts):
            self.point_id_field[i] = i

        # Create root node with all points
        self.node_field[0].prim_count = self.npts
        self.node_field[0].left_first = 0
        self._update_node_bounds(0)

        # Iteratively subdivide nodes
        self._iterative_node_subdivide()

    @ti.kernel
    def nearest_neighbor_kernel(self, pt: tm.vec3) -> ti.i32:
        return self.nearest_neighbor(pt)

    @ti.func
    def nearest_neighbor(self, pt: tm.vec3) -> ti.i32:
        # Initialize return variables
        cp_idx = 0
        dmin = tm.inf

        # Initialize stacks
        node_stack = ivec32(0)
        lr_stack = ivec32(0)
        node_stack_index = 0
        lr_stack_index = -1

        # Initialize current node id
        node_idx = -1

        # Process BVH
        while node_stack_index > -1:
            # Pop next node to process
            if node_idx == -1:
                # Pop root node
                node_idx = node_stack[node_stack_index]
                node_stack_index -= 1
            else:
                # Process sister of previous node

                # Pop parent node from stack
                parent_idx = node_stack[node_stack_index]
                node_stack_index -= 1

                # Pop relation to parent node
                prev_lr = lr_stack[lr_stack_index]
                lr_stack_index -= 1

                # Get yet to be processed child node
                left_idx = self.node_field[parent_idx].left_first
                right_idx = left_idx + 1
                node_idx = left_idx if (prev_lr == 1) else right_idx

            # Proceed depth-wise until leaf or node cannot contain closer primitive
            branch_searched = False
            while not branch_searched:
                # Only process nodes which may contain closer primitive
                if sphere_aabb_intersection(
                    pt,
                    dmin,
                    self.node_field[node_idx].aabb_min,
                    self.node_field[node_idx].aabb_max,
                ):
                    # Process leaf node
                    if self.node_field[node_idx].is_leaf():
                        # Search leaf node primitives
                        i = self.node_field[node_idx].left_first
                        prim_count = self.node_field[node_idx].prim_count
                        for offset in range(prim_count):
                            leaf_prim_idx = i + offset

                            if self.is_point_removed[
                                self.point_id_field[leaf_prim_idx]
                            ]:
                                continue

                            prim = self.points[self.point_id_field[leaf_prim_idx]]
                            d = tm.length(prim - pt)
                            if d < dmin:
                                dmin = d
                                cp_idx = self.point_id_field[leaf_prim_idx]

                        # Exit loop
                        branch_searched = True

                    else:
                        # Push current node to stack
                        node_stack_index += 1
                        node_stack[node_stack_index] = node_idx

                        # Proceed to best candidate child
                        left_idx = self.node_field[node_idx].left_first
                        right_idx = left_idx + 1
                        if aabb_contains(
                            self.node_field[left_idx].aabb_min,
                            self.node_field[left_idx].aabb_max,
                            pt,
                        ):
                            node_idx = left_idx
                        elif aabb_contains(
                            self.node_field[right_idx].aabb_min,
                            self.node_field[right_idx].aabb_max,
                            pt,
                        ):
                            node_idx = right_idx
                        else:
                            # Select node whose centroid is closer to target point
                            left_dist = square_distance_point_to_aabb(
                                pt,
                                self.node_field[left_idx].aabb_min,
                                self.node_field[left_idx].aabb_max,
                            )
                            right_dist = square_distance_point_to_aabb(
                                pt,
                                self.node_field[right_idx].aabb_min,
                                self.node_field[right_idx].aabb_max,
                            )
                            node_idx = (
                                left_idx if (left_dist < right_dist) else right_idx
                            )

                        # Keep track of path from parent to child
                        lr_stack_index += 1
                        lr_stack[lr_stack_index] = node_idx == right_idx

                else:
                    # Exit loop
                    branch_searched = True

        return cp_idx

    @ti.kernel
    def points_in_sphere_kernel(
        self, sphere_center: tm.vec3, sphere_radius: ti.f32, stack: ti.template()
    ):
        self.points_in_sphere(sphere_center, sphere_radius, stack)

    @ti.func
    def points_in_sphere(
        self, sphere_center: tm.vec3, sphere_radius: ti.f32, stack: ti.template()
    ):
        stack.clear()

        # Initialize stacks
        node_stack = ivec32(0)
        lr_stack = ivec32(0)
        node_stack_index = 0
        lr_stack_index = -1

        # Initialize current node id
        node_idx = -1

        # Process BVH
        while node_stack_index > -1:
            # Pop next node to process
            if node_idx == -1:
                # Pop root node
                node_idx = node_stack[node_stack_index]
                node_stack_index -= 1
            else:
                # Process sister of previous node

                # Pop parent node from stack
                parent_idx = node_stack[node_stack_index]
                node_stack_index -= 1

                # Pop relation to parent node
                prev_lr = lr_stack[lr_stack_index]
                lr_stack_index -= 1

                # Get yet to be processed child node
                left_idx = self.node_field[parent_idx].left_first
                right_idx = left_idx + 1
                node_idx = left_idx if (prev_lr == 1) else right_idx

            # Proceed depth-wise until leaf or node cannot contain closer primitive
            branch_searched = False
            while not branch_searched:
                # Only process nodes which may contain closer primitive
                if sphere_aabb_intersection(
                    sphere_center,
                    sphere_radius,
                    self.node_field[node_idx].aabb_min,
                    self.node_field[node_idx].aabb_max,
                ):
                    # Process leaf node
                    if self.node_field[node_idx].is_leaf():
                        # Search leaf node primitives
                        i = self.node_field[node_idx].left_first
                        prim_count = self.node_field[node_idx].prim_count
                        for offset in range(prim_count):
                            leaf_prim_idx = i + offset

                            if self.is_point_removed[
                                self.point_id_field[leaf_prim_idx]
                            ]:
                                continue

                            prim = self.points[self.point_id_field[leaf_prim_idx]]
                            d = tm.length(prim - sphere_center)
                            if d < sphere_radius:
                                stack.push(self.point_id_field[leaf_prim_idx])

                        # Exit loop
                        branch_searched = True

                    else:
                        # Push current node to stack
                        node_stack_index += 1
                        node_stack[node_stack_index] = node_idx

                        # Proceed to best candidate child
                        left_idx = self.node_field[node_idx].left_first
                        right_idx = left_idx + 1
                        if aabb_contains(
                            self.node_field[left_idx].aabb_min
                            - sphere_radius * tm.vec3(1),
                            self.node_field[left_idx].aabb_max
                            + sphere_radius * tm.vec3(1),
                            sphere_center,
                        ):
                            node_idx = left_idx
                        elif aabb_contains(
                            self.node_field[right_idx].aabb_min
                            - sphere_radius * tm.vec3(1),
                            self.node_field[right_idx].aabb_max
                            + sphere_radius * tm.vec3(1),
                            sphere_center,
                        ):
                            node_idx = right_idx
                        else:
                            # Select node whose centroid is closer to target point
                            left_dist = square_distance_point_to_aabb(
                                sphere_center,
                                self.node_field[left_idx].aabb_min,
                                self.node_field[left_idx].aabb_max,
                            )
                            right_dist = square_distance_point_to_aabb(
                                sphere_center,
                                self.node_field[right_idx].aabb_min,
                                self.node_field[right_idx].aabb_max,
                            )
                            node_idx = (
                                left_idx if (left_dist < right_dist) else right_idx
                            )

                        # Keep track of path from parent to child
                        lr_stack_index += 1
                        lr_stack[lr_stack_index] = node_idx == right_idx

                else:
                    # Exit loop
                    branch_searched = True

    @ti.kernel
    def points_in_sphere_local_stack_kernel(
        self, sphere_center: tm.vec3, sphere_radius: ti.f32
    ) -> ivecn:
        return self.points_in_sphere_local_stack(sphere_center, sphere_radius)

    @ti.func
    def points_in_sphere_local_stack(
        self, sphere_center: tm.vec3, sphere_radius: ti.f32
    ) -> ivecn:
        stack = ivecn(-1)
        stack_index = -1

        # Initialize stacks
        node_stack = ivec32(0)
        lr_stack = ivec32(0)
        node_stack_index = 0
        lr_stack_index = -1

        # Initialize current node id
        node_idx = -1

        # Process BVH
        while node_stack_index > -1:
            # Pop next node to process
            if node_idx == -1:
                # Pop root node
                node_idx = node_stack[node_stack_index]
                node_stack_index -= 1
            else:
                # Process sister of previous node

                # Pop parent node from stack
                parent_idx = node_stack[node_stack_index]
                node_stack_index -= 1

                # Pop relation to parent node
                prev_lr = lr_stack[lr_stack_index]
                lr_stack_index -= 1

                # Get yet to be processed child node
                left_idx = self.node_field[parent_idx].left_first
                right_idx = left_idx + 1
                node_idx = left_idx if (prev_lr == 1) else right_idx

            # Proceed depth-wise until leaf or node cannot contain closer primitive
            branch_searched = False
            while not branch_searched:
                # Only process nodes which may contain closer primitive
                if sphere_aabb_intersection(
                    sphere_center,
                    sphere_radius,
                    self.node_field[node_idx].aabb_min,
                    self.node_field[node_idx].aabb_max,
                ):
                    # Process leaf node
                    if self.node_field[node_idx].is_leaf():
                        # Search leaf node primitives
                        i = self.node_field[node_idx].left_first
                        prim_count = self.node_field[node_idx].prim_count
                        for offset in range(prim_count):
                            leaf_prim_idx = i + offset

                            if self.is_point_removed[
                                self.point_id_field[leaf_prim_idx]
                            ]:
                                continue

                            prim = self.points[self.point_id_field[leaf_prim_idx]]
                            d = tm.length(prim - sphere_center)
                            if d < sphere_radius:
                                # stack.push(self.point_id_field[leaf_prim_idx])
                                stack_index += 1
                                stack[stack_index] = self.point_id_field[leaf_prim_idx]

                        # Exit loop
                        branch_searched = True

                    else:
                        # Push current node to stack
                        node_stack_index += 1
                        node_stack[node_stack_index] = node_idx

                        # Proceed to best candidate child
                        left_idx = self.node_field[node_idx].left_first
                        right_idx = left_idx + 1
                        if aabb_contains(
                            self.node_field[left_idx].aabb_min
                            - sphere_radius * tm.vec3(1),
                            self.node_field[left_idx].aabb_max
                            + sphere_radius * tm.vec3(1),
                            sphere_center,
                        ):
                            node_idx = left_idx
                        elif aabb_contains(
                            self.node_field[right_idx].aabb_min
                            - sphere_radius * tm.vec3(1),
                            self.node_field[right_idx].aabb_max
                            + sphere_radius * tm.vec3(1),
                            sphere_center,
                        ):
                            node_idx = right_idx
                        else:
                            # Select node whose centroid is closer to target point
                            left_dist = square_distance_point_to_aabb(
                                sphere_center,
                                self.node_field[left_idx].aabb_min,
                                self.node_field[left_idx].aabb_max,
                            )
                            right_dist = square_distance_point_to_aabb(
                                sphere_center,
                                self.node_field[right_idx].aabb_min,
                                self.node_field[right_idx].aabb_max,
                            )
                            node_idx = (
                                left_idx if (left_dist < right_dist) else right_idx
                            )

                        # Keep track of path from parent to child
                        lr_stack_index += 1
                        lr_stack[lr_stack_index] = node_idx == right_idx

                else:
                    # Exit loop
                    branch_searched = True

        return stack

    @ti.kernel
    def any_pt_in_cone_kernel(
        self, cone_origin: tm.vec3, cone_direction: tm.vec3, cone_angle: ti.f32
    ) -> ti.i32:
        return self.any_pt_in_cone(cone_origin, cone_direction, cone_angle)

    @ti.func
    def any_pt_in_cone(
        self, cone_origin: tm.vec3, cone_direction: tm.vec3, cone_angle: ti.f32
    ) -> ti.i32:
        # Initialize return bool
        is_found = 0

        # Reused variable
        cos_half_angle = tm.cos(cone_angle)

        # Initialize stacks
        node_stack = ivec32(0)
        lr_stack = ivec32(0)
        node_stack_index = 0
        lr_stack_index = -1

        # Initialize current node id
        node_idx = -1

        # Process BVH
        while (node_stack_index > -1) and not is_found:
            # Pop next node to process
            if node_idx == -1:
                # Pop root node
                node_idx = node_stack[node_stack_index]
                node_stack_index -= 1
            else:
                # Process sister of previous node

                # Pop parent node from stack
                parent_idx = node_stack[node_stack_index]
                node_stack_index -= 1

                # Pop relation to parent node
                prev_lr = lr_stack[lr_stack_index]
                lr_stack_index -= 1

                # Get yet to be processed child node
                left_idx = self.node_field[parent_idx].left_first
                right_idx = left_idx + 1
                node_idx = left_idx if (prev_lr == 1) else right_idx

            # Proceed depth-wise until leaf or node cannot contain closer primitive
            branch_searched = False
            while not branch_searched and not is_found:
                # Only process nodes which may contain closer primitive
                if aabb_cone_possible_intersection(
                    self.node_field[node_idx].aabb_min,
                    self.node_field[node_idx].aabb_max,
                    cone_origin,
                    cone_direction,
                    cone_angle,
                ):
                    # Process leaf node
                    if self.node_field[node_idx].is_leaf():
                        # Search leaf node primitives
                        i = self.node_field[node_idx].left_first
                        prim_count = self.node_field[node_idx].prim_count
                        for offset in range(prim_count):
                            leaf_prim_idx = i + offset
                            prim = self.points[self.point_id_field[leaf_prim_idx]]

                            if self.is_point_removed[
                                self.point_id_field[leaf_prim_idx]
                            ]:
                                continue

                            if point_in_cone(
                                prim, cone_origin, cone_direction, cos_half_angle
                            ):
                                is_found = 1

                        # Exit loop
                        branch_searched = True

                    else:
                        # Push current node to stack
                        node_stack_index += 1
                        node_stack[node_stack_index] = node_idx

                        # Proceed to best candidate child
                        left_idx = self.node_field[node_idx].left_first
                        right_idx = left_idx + 1

                        left_aabb_cntr = 0.5 * (
                            self.node_field[left_idx].aabb_min
                            + self.node_field[left_idx].aabb_max
                        )
                        right_aabb_cntr = 0.5 * (
                            self.node_field[right_idx].aabb_min
                            + self.node_field[right_idx].aabb_max
                        )

                        d_left = square_length(left_aabb_cntr - cone_origin)
                        d_right = square_length(right_aabb_cntr - cone_origin)

                        if d_left < d_right:
                            node_idx = left_idx
                        else:
                            node_idx = right_idx

                        if not branch_searched:
                            # Keep track of path from parent to child
                            lr_stack_index += 1
                            lr_stack[lr_stack_index] = node_idx == right_idx

                else:
                    # Exit loop
                    branch_searched = True

        return is_found

    @ti.kernel
    def _build_kernel(self):
        # Initialize point id field
        for i in range(self.npts):
            self.point_id_field[i] = i

        # Create root node with all points
        self.node_field[0].prim_count = self.npts
        self.node_field[0].left_first = 0
        self._update_node_bounds(0)

        # Iteratively subdivide nodes
        self._iterative_node_subdivide()

    @ti.func
    def _update_node_bounds(self, node_idx):
        # Reset box extents
        self.node_field[node_idx].aabb_min = tm.vec3(tm.inf)
        self.node_field[node_idx].aabb_max = tm.vec3(-tm.inf)

        # Loop over all points in box
        ti.loop_config(serialize=True)
        for offset in range(self.node_field[node_idx].prim_count):
            leaf_prim_idx = self.node_field[node_idx].left_first + offset

            if self.is_point_removed[self.point_id_field[leaf_prim_idx]]:
                continue

            pt = self.points[self.point_id_field[leaf_prim_idx]]

            # Loop over all dimensions
            for i in ti.static(range(3)):
                # Update box extents to include point
                self.node_field[node_idx].aabb_min[i] = min(
                    self.node_field[node_idx].aabb_min[i], pt[i]
                )
                self.node_field[node_idx].aabb_max[i] = max(
                    self.node_field[node_idx].aabb_max[i], pt[i]
                )

    @ti.func
    def _iterative_node_subdivide(self):
        # Keep track of node count for node identification
        nodes_used = 1

        # Stack keeps track of nodes to process
        stack = ivec32(0)
        stack_idx = 0  # index of top of stack (stack initialized as [0] here)

        while stack_idx >= 0:
            # Process next node and remove from stack
            node_idx = stack[stack_idx]
            stack_idx -= 1

            # Terminate recursion if less primitives than max allowed
            true_prim_count = 0
            for offset in range(self.node_field[node_idx].prim_count):
                if not self.is_point_removed[
                    self.point_id_field[self.node_field[node_idx].left_first + offset]
                ]:
                    true_prim_count += 1

            if true_prim_count > MAX_PRIM_PER_NODE:
                # Split at box median along longest extent
                extent = (
                    self.node_field[node_idx].aabb_max
                    - self.node_field[node_idx].aabb_min
                )
                axis = 0
                if extent.y > extent.x:
                    axis = 1
                if extent.z > extent[axis]:
                    axis = 2
                split_pos = (
                    self.node_field[node_idx].aabb_min[axis] + extent[axis] * 0.5
                )

                # QuickSort primitives along split dimension
                i = self.node_field[node_idx].left_first
                j = i + self.node_field[node_idx].prim_count - 1
                while i <= j:
                    if self.points[self.point_id_field[i]][axis] < split_pos:
                        i += 1
                    else:
                        tmp = self.point_id_field[i]
                        self.point_id_field[i] = self.point_id_field[j]
                        self.point_id_field[j] = tmp
                        j -= 1

                # Abort split if one of the sides is empty
                left_count = i - self.node_field[node_idx].left_first

                if not (
                    left_count == 0
                    or left_count == self.node_field[node_idx].prim_count
                ):
                    # Create left child and push to stack
                    left_child_idx = nodes_used
                    nodes_used += 1
                    self.node_field[left_child_idx].left_first = self.node_field[
                        node_idx
                    ].left_first
                    self.node_field[left_child_idx].prim_count = left_count
                    self.node_field[node_idx].left_first = left_child_idx
                    self._update_node_bounds(left_child_idx)

                    # Create right child and push to stack
                    right_child_idx = nodes_used
                    nodes_used += 1
                    self.node_field[right_child_idx].left_first = i
                    self.node_field[right_child_idx].prim_count = (
                        self.node_field[node_idx].prim_count - left_count
                    )
                    self.node_field[node_idx].prim_count = 0
                    self._update_node_bounds(right_child_idx)

                    # Push right child to stack first
                    stack_idx += 1
                    stack[stack_idx] = right_child_idx

                    # Push left child to stack
                    stack_idx += 1
                    stack[stack_idx] = left_child_idx


@ti.dataclass
class BVHNode:
    aabb_min: tm.vec3
    aabb_max: tm.vec3
    left_first: ti.int32
    prim_count: ti.int32

    @ti.func
    def is_leaf(self):
        return self.prim_count > 0

    def print(self):
        print("aabb min: {}".format(self.aabb_min))
        print("aabb max: {}".format(self.aabb_max))
        if self.prim_count == 0:
            print("left child idx: {}".format(self.left_first))
        else:
            print("first prim idx: {}".format(self.left_first))
        print("prim count: {}".format(self.prim_count))


# General functions
@ti.kernel
def kernel_copy(from_field: ti.template(), to_field: ti.template()):
    for I in ti.grouped(from_field):
        to_field[I] = from_field[I]


@ti.kernel
def kernel_reorder(
    input_field: ti.template(), indices: ti.template(), output_field: ti.template()
):
    for i in input_field:
        output_field[i] = input_field[indices[i]]


@ti.func
def square_length(vec: tm.vec3):
    return vec[0] ** 2 + vec[1] ** 2 + vec[2] ** 2


# AABB helper functions
@ti.func
def square_distance_point_to_aabb(
    pt: tm.vec3, aabb_min: tm.vec3, aabb_max: tm.vec3
) -> ti.float32:
    closest_point = tm.vec3(0, 0, 0)

    # Find closest point on AABB for each dimension
    ti.loop_config(serialize=True)
    for i in ti.static(range(3)):
        if pt[i] < aabb_min[i]:
            closest_point[i] = aabb_min[i]
        elif pt[i] > aabb_max[i]:
            closest_point[i] = aabb_max[i]
        else:
            closest_point[i] = pt[i]

    # Calculate distance using Euclidean distance formula
    return (
        (pt[0] - closest_point[0]) ** 2
        + (pt[1] - closest_point[1]) ** 2
        + (pt[2] - closest_point[2]) ** 2
    )


@ti.func
def aabb_contains(aabb_min: tm.vec3, aabb_max: tm.vec3, pt: tm.vec3):
    is_contained = True
    ti.loop_config(serialize=True)
    for i in ti.static(range(3)):
        if not ((pt[i] >= aabb_min[i]) and (pt[i] <= aabb_max[i])):
            is_contained = False
    return is_contained


@ti.func
def sphere_aabb_intersection(
    sphere_center: tm.vec3,
    sphere_radius: ti.float32,
    aabb_min: tm.vec3,
    aabb_max: tm.vec3,
):
    # find the squared distance between the sphere center and the AABB
    sq_dist = 0.0
    ti.loop_config(serialize=True)
    for i in ti.static(range(3)):
        if sphere_center[i] < aabb_min[i]:
            sq_dist += (aabb_min[i] - sphere_center[i]) ** 2
        elif sphere_center[i] > aabb_max[i]:
            sq_dist += (sphere_center[i] - aabb_max[i]) ** 2

    # check if the squared distance is less than the squared radius of the sphere
    return sq_dist <= sphere_radius**2


@ti.func
def aabb_cone_possible_intersection(
    aabb_min: tm.vec3, aabb_max: tm.vec3, cone_origin, cone_direction, cone_angle
) -> ti.int32:
    # Initialize return
    ret = 0

    # Calculate aabb bounding sphere
    c = 0.5 * (aabb_min + aabb_max)
    r = 0.5 * tm.length(aabb_max - aabb_min)

    # Offset cone
    offset_cone_origin = cone_origin - r / tm.sin(cone_angle) * cone_direction

    # Check if sphere center inside offset cone
    op = c - offset_cone_origin
    op_length = tm.length(op)
    if op_length > FLOAT_EPSILON:
        op_normalized = op / op_length
        cos_theta = tm.dot(op_normalized, cone_direction)

        if not (cos_theta <= tm.cos(cone_angle) or cos_theta <= 0.0):
            ret = 1

    return ret


@ti.func
def point_in_cone(p: tm.vec3, o: tm.vec3, d: tm.vec3, cos_half_angle: float):
    """
    Assume d is normalized
    """

    # vector from origin to point
    op = p - o
    op_length = tm.length(op)
    is_inside = True
    if op_length > FLOAT_EPSILON:
        op_normalized = op / op_length
        cos_theta = tm.dot(op_normalized, d)

        if cos_theta <= cos_half_angle or cos_theta <= 0.0:
            is_inside = False
    else:
        is_inside = False

    return is_inside


@ti.func
def sum_field(field: ti.template()) -> ti.f32:
    sum = 0.0
    for i in range(field.shape[0]):
        sum += field[i]
    return sum
