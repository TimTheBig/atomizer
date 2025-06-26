import argparse

import matplotlib
import numpy as np
from tqdm import tqdm

import atom.toolpath3


def from_to_rotation(dir, up):
    dir /= np.linalg.norm(dir)
    x = np.cross(up, dir)
    if np.linalg.norm(x) < 0.001:
        x = [1, 0, 0]
    x /= np.linalg.norm(x)
    y = np.cross(dir, x)
    y /= np.linalg.norm(y)
    return np.array((x, y, dir))


def orientation_to_normal(d):
    theta = d[0]
    phi = d[1]
    return np.array(
        (np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta))
    )


def is_deposition(path, i):
    return (
        i < path.point_count
        and path.travel_type[i] == atom.toolpath3.TRAVEL_TYPE_DEPOSITION
        and np.linalg.norm(path.point[i] - path.point[i - 1]) > 0.0001
    )


def normalize(v):
    return v / np.linalg.norm(v)


def path_to_ply(input, output, color):
    path = atom.toolpath3.Toolpath()
    path.load(input)
    vertex = []
    face = []

    path_length = 0
    full_path_length = 0
    for i in tqdm(range(path.point_count)):
        if is_deposition(path, i):
            full_path_length += np.linalg.norm(path.point[i] - path.point[i - 1])
    for i in tqdm(range(path.point_count)):
        if is_deposition(path, i):
            start = len(vertex)
            k = 1 / np.sqrt(2)
            path_length += np.linalg.norm(path.point[i] - path.point[i - 1])
            if color == "progress":
                C0 = [
                    int(255 * c)
                    for c in matplotlib.cm.turbo(path_length / full_path_length)
                ]
            else:
                direction = (
                    np.arctan(
                        (path.point[i][1] - path.point[i - 1][1])
                        / (path.point[i][0] - path.point[i - 1][0])
                    )
                    if path.point[i][0] - path.point[i - 1][0] != 0
                    else np.pi * 0.5
                )
                C0 = [
                    int(255 * c)
                    for c in matplotlib.cm.twilight(direction / np.pi + 0.5)
                ]

            V = [
                np.array((1, 1, 0)),
                np.array((1 + k, k, 0)),
                np.array((2, 0, 0)),
                np.array((1 + k, -k, 0)),
                np.array((1, -1, 0)),
                np.array((-1, -1, 0)),
                np.array((-1 - k, -k, 0)),
                np.array((-2, 0, 0)),
                np.array((-1 - k, k, 0)),
                np.array((-1, 1, 0)),
            ]

            S0 = np.array(
                ((1, 0, 0), (0, 2.0 * path.height[i] / path.width[i], 0), (0, 0, 1))
            )
            R0 = (
                from_to_rotation(
                    path.point[i] - path.point[i - 1],
                    orientation_to_normal(path.tool_orientation[i]),
                ).transpose()
                @ S0
            )
            V0 = []
            for v in V:
                V0.append(R0 @ v * path.width[i] * 0.25)

            eta = 0.5
            p0 = path.point[i - 1] * (1 - eta) + path.point[i] * eta
            p1 = path.point[i - 1] * eta + path.point[i] * (1 - eta)
            p0 -= (
                normalize(
                    orientation_to_normal(path.tool_orientation[i - 1]) * (1 - eta)
                    + orientation_to_normal(path.tool_orientation[i]) * eta
                )
                * 0.5
                * (path.height[i - 1] * (1 - eta) + path.height[i] * eta)
            )
            p1 -= (
                normalize(
                    orientation_to_normal(path.tool_orientation[i - 1]) * eta
                    + orientation_to_normal(path.tool_orientation[i]) * (1 - eta)
                )
                * 0.5
                * (path.height[i - 1] * eta + path.height[i] * (1 - eta))
            )

            for v in V0:
                vertex.append((v + p0, C0))
            for v in V0:
                vertex.append((v + p1, C0))

            l = len(V)
            for j in range(l):
                face.append(
                    [
                        start + j,
                        start + ((j + 1) % l),
                        start + l + ((j + 1) % l),
                        start + l + j,
                    ]
                )

            if is_deposition(path, i + 1):
                if color == "progress":
                    C1 = [
                        int(255 * c)
                        for c in matplotlib.cm.turbo(
                            (
                                path_length
                                + np.linalg.norm(path.point[i + 1] - path.point[i])
                            )
                            / full_path_length
                        )
                    ]
                else:
                    direction = (
                        np.arctan(
                            (path.point[i + 1][1] - path.point[i][1])
                            / (path.point[i + 1][0] - path.point[i][0])
                        )
                        if path.point[i + 1][0] - path.point[i][0] != 0
                        else np.pi * 0.5
                    )
                    C1 = [
                        int(255 * c)
                        for c in matplotlib.cm.twilight(direction / np.pi + 0.5)
                    ]

                S1 = np.array(
                    (
                        (1, 0, 0),
                        (0, 2.0 * path.height[i + 1] / path.width[i + 1], 0),
                        (0, 0, 1),
                    )
                )
                R1 = (
                    from_to_rotation(
                        path.point[i + 1] - path.point[i],
                        orientation_to_normal(path.tool_orientation[i + 1]),
                    ).transpose()
                    @ S1
                )
                V1 = []
                for v in V:
                    V1.append(R1 @ v * path.width[i + 1] * 0.25)

                p2 = path.point[i + 1] * eta + path.point[i] * (1 - eta)
                p2 -= (
                    normalize(
                        orientation_to_normal(path.tool_orientation[i + 1]) * eta
                        + orientation_to_normal(path.tool_orientation[i]) * (1 - eta)
                    )
                    * 0.5
                    * (path.height[i + 1] * eta + path.height[i] * (1 - eta))
                )
                for v in V1:
                    vertex.append((v + p2, C1))
                for j in range(l):
                    face.append(
                        [
                            start + j + l,
                            start + ((j + 1) % l) + l,
                            start + 2 * l + ((j + 1) % l),
                            start + 2 * l + j,
                        ]
                    )

            if (
                not is_deposition(path, i + 1)
                or np.dot(
                    path.point[i + 1] - path.point[i], path.point[i] - path.point[i - 1]
                )
                < 0
            ):
                s = len(vertex)
                for v in V:
                    vertex.append(
                        (R0 @ ((v + [0, 0, 1.8]) * path.width[i] * 0.18) + p1, C0)
                    )
                vertex.append(
                    (R0 @ (np.array((0, 0, 2)) * path.width[i] * 0.25) + p1, C0)
                )
                for j in range(l):
                    face.append([s + j, s + (j + 1) % l, s + l])
                    face.append(
                        [
                            start + j + l,
                            start + ((j + 1) % l) + l,
                            s + ((j + 1) % l),
                            s + j,
                        ]
                    )

            if not is_deposition(path, i - 1):
                s = len(vertex)
                for v in V:
                    vertex.append(
                        (R0 @ ((v + [0, 0, -1.8]) * path.width[i] * 0.18) + p0, C0)
                    )
                vertex.append(
                    (R0 @ (np.array((0, 0, -2)) * path.width[i] * 0.25) + p0, C0)
                )
                for j in range(l):
                    face.append([s + j, s + (j + 1) % l, s + l])
                    face.append(
                        [start + j, start + ((j + 1) % l), s + ((j + 1) % l), s + j]
                    )

    with open(output, "w") as f:
        f.write(
            f"ply\nformat ascii 1.0\nelement vertex {len(vertex)}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nproperty uchar alpha\nelement face {len(face)}\nproperty list uchar uint vertex_indices\nend_header\n"
        )

        for v in tqdm(vertex):
            f.write(
                f"{v[0][0]} {v[0][1]} {v[0][2]} {v[1][0]} {v[1][1]} {v[1][2]} {v[1][3]}\n"
            )
        for fa in tqdm(face):
            f.write(f"{len(fa)} ")
            for i in fa:
                f.write(f"{i} ")
            f.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate an anisotropic tubes based on an input toolpath."
    )
    parser.add_argument("toolpath_path")
    parser.add_argument("mesh_path", help="The path to the output PLY mesh.")
    parser.add_argument("color", nargs="?", default="progress")
    args = parser.parse_args()

    path_to_ply(
        args.toolpath_path,
        args.mesh_path,
        args.color,
    )
