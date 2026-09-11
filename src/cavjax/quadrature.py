"""Fixed positive vertex quadratures on frequency-subdivided icosahedra."""

from __future__ import annotations

from dataclasses import dataclass
import numbers

import numpy as np


@dataclass(frozen=True)
class ReferenceMesh:
    """A fixed spherical reference chart.

    ``directions`` are unit vectors and ``weights`` are positive solid angles
    summing to 4π. Molecular geometry never changes this topology.
    """

    vertices: np.ndarray
    faces: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    frequency: int


_VERTICES = np.array([
    [-0.5257311121191336, 0.85065080835204, 0.0],
    [0.5257311121191336, 0.85065080835204, 0.0],
    [-0.5257311121191336, -0.85065080835204, 0.0],
    [0.5257311121191336, -0.85065080835204, 0.0],
    [0.0, -0.5257311121191336, 0.85065080835204],
    [0.0, 0.5257311121191336, 0.85065080835204],
    [0.0, -0.5257311121191336, -0.85065080835204],
    [0.0, 0.5257311121191336, -0.85065080835204],
    [0.85065080835204, 0.0, -0.5257311121191336],
    [0.85065080835204, 0.0, 0.5257311121191336],
    [-0.85065080835204, 0.0, -0.5257311121191336],
    [-0.85065080835204, 0.0, 0.5257311121191336],
], dtype=float)

_FACES = np.array([
    [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
    [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
    [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
    [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
], dtype=np.int32)


def frequency_for_target(target_points: float) -> int:
    """Return the frequency whose ``10*f**2 + 2`` count is nearest the target."""
    if not np.isfinite(target_points) or target_points <= 0:
        raise ValueError("target_points must be finite and positive")
    low = max(1, int(np.floor(np.sqrt(max(0.0, target_points - 2.0) / 10.0))))
    return min((low, low + 1), key=lambda f: (abs(10 * f * f + 2 - target_points), f))


def vertex_count(frequency: int) -> int:
    if not isinstance(frequency, numbers.Integral) or frequency < 1:
        raise ValueError("frequency must be a positive integer")
    return 10 * int(frequency) ** 2 + 2


def frequency_mesh(frequency: int) -> ReferenceMesh:
    """Build a deduplicated frequency-subdivided icosahedral vertex chart."""
    if not isinstance(frequency, numbers.Integral) or frequency < 1:
        raise ValueError("frequency must be a positive integer")
    frequency = int(frequency)
    vertices: list[np.ndarray] = []
    faces: list[list[int]] = []
    lookup: dict[tuple[float, float, float], int] = {}

    for face in _FACES:
        a, b, c = _VERTICES[face]
        local: dict[tuple[int, int], int] = {}
        for i in range(frequency + 1):
            for j in range(frequency - i + 1):
                point = (frequency - i - j) * a + i * b + j * c
                point /= np.linalg.norm(point)
                key = tuple(np.round(point, 12))
                if key not in lookup:
                    lookup[key] = len(vertices)
                    vertices.append(point)
                local[i, j] = lookup[key]
        for i in range(frequency):
            for j in range(frequency - i):
                faces.append([local[i, j], local[i + 1, j], local[i, j + 1]])
                if i + j < frequency - 1:
                    faces.append([local[i + 1, j], local[i + 1, j + 1], local[i, j + 1]])

    vertices_array = np.asarray(vertices)
    faces_array = np.asarray(faces, dtype=np.int32)
    if len(vertices_array) != vertex_count(frequency):
        raise RuntimeError("frequency mesh has incorrect vertex count")
    if len(faces_array) != 20 * frequency**2:
        raise RuntimeError("frequency mesh has incorrect face count")

    triangles = vertices_array[faces_array]
    a, b, c = triangles.transpose(1, 0, 2)
    numerator = np.abs(np.einsum("ij,ij->i", a, np.cross(b, c)))
    denominator = 1 + (a * b).sum(1) + (b * c).sum(1) + (c * a).sum(1)
    triangle_weights = 2 * np.arctan2(numerator, denominator)
    weights = np.zeros(len(vertices_array))
    np.add.at(weights, faces_array.ravel(), np.repeat(triangle_weights / 3, 3))
    if np.any(weights <= 0) or not np.isclose(weights.sum(), 4 * np.pi, atol=1e-12):
        raise RuntimeError("invalid spherical quadrature")

    return ReferenceMesh(
        vertices=vertices_array,
        faces=faces_array,
        directions=vertices_array.copy(),
        weights=weights,
        frequency=frequency,
    )


def mesh_for_target(target_points: float) -> ReferenceMesh:
    return frequency_mesh(frequency_for_target(target_points))
