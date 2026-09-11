"""Smooth convex cavities and contracted geometry response.

All lengths are in Bohr, areas in Bohr², and normals are outward unit vectors.
The module is independent of molecular file formats and electronic-structure
packages: callers provide Cartesian positions and positive atomic radii.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import numpy as np

from .quadrature import ReferenceMesh, mesh_for_target


class Surface(NamedTuple):
    points: jax.Array
    areas: jax.Array
    normals: jax.Array


class SurfaceCotangent(NamedTuple):
    points: jax.Array
    areas: jax.Array
    normals: jax.Array


@dataclass(frozen=True)
class CavityConfig:
    """Numerical definition and resolution of a cavity.

    Parameters are Bohr-native. ``target_points`` and
    ``target_shape_directions`` are rounded to the nearest valid count
    ``10*f²+2`` of a frequency-subdivided icosahedral vertex chart.
    """

    target_points: float
    target_shape_directions: float = 162
    support_temperature: float = 0.2834589186938655
    wall_temperature: float = 0.15117808997006163
    root_steps: int = 48

    def __post_init__(self) -> None:
        for name in ("target_points", "target_shape_directions",
                     "support_temperature", "wall_temperature"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.root_steps, int) or self.root_steps < 32:
            raise ValueError("root_steps must be an integer >= 32")


def _validate_x64() -> None:
    if not jax.config.jax_enable_x64:
        raise RuntimeError(
            "CavJAX requires JAX float64. Set JAX_ENABLE_X64=1 before starting "
            "Python, or call jax.config.update('jax_enable_x64', True) before "
            "constructing a Cavity. CavJAX does not mutate global JAX settings."
        )


def _validate_atoms(positions, radii) -> tuple[np.ndarray, np.ndarray]:
    positions_array = np.asarray(positions)
    radii_array = np.asarray(radii)
    if positions_array.ndim != 2 or positions_array.shape[1:] != (3,) or len(positions_array) == 0:
        raise ValueError("positions must have nonempty shape (N, 3)")
    if radii_array.shape != (len(positions_array),):
        raise ValueError("radii must have shape (N,)")
    if not np.all(np.isfinite(positions_array)):
        raise ValueError("positions must be finite")
    if not np.all(np.isfinite(radii_array)) or np.any(radii_array <= 0):
        raise ValueError("radii must be finite and positive")
    return positions_array, radii_array


def _soft_support(centered, radii, directions, temperature):
    return temperature * logsumexp(
        (directions @ centered.T + radii) / temperature, axis=1
    )


def _sampling_map(centered, radii):
    """Smooth rotationally covariant affine preconditioner without PCA."""
    covariance = centered.T @ centered / len(centered)
    matrix = covariance + jnp.mean(radii) ** 2 * jnp.eye(3)
    scale = jnp.trace(matrix)
    y, z = matrix / scale, jnp.eye(3)

    def step(_, state):
        y, z = state
        transform = (3 * jnp.eye(3) - z @ y) / 2
        return y @ transform, transform @ z

    y, _ = jax.lax.fori_loop(0, 12, step, (y, z))
    return jnp.sqrt(scale) * (y + y.T) / 2


def _map_directions(centered, radii, directions, weights):
    transform = _sampling_map(centered, radii)
    mapped = directions @ transform.T
    lengths = jnp.linalg.norm(mapped, axis=1)
    angular_jacobian = jnp.linalg.det(transform) / lengths**3
    return mapped / lengths[:, None], weights * angular_jacobian


def _implicit_field(points_relative, offsets, support_directions, wall_temperature):
    scores = (
        points_relative @ support_directions.T - offsets
    ) / wall_temperature
    return wall_temperature * (
        logsumexp(scores, axis=1) - jnp.log(len(support_directions))
    )


def _radial_roots(directions, offsets, support_directions, wall_temperature, steps):
    projections = directions @ support_directions.T

    def field(distance):
        scores = (
            distance[:, None] * projections - offsets
        ) / wall_temperature
        return wall_temperature * (
            logsumexp(scores, axis=1) - jnp.log(len(support_directions))
        )

    upper = (
        jnp.max(offsets)
        + wall_temperature * jnp.log(len(support_directions))
        + 1.0
    ) / jnp.max(projections, axis=1)

    def solve(function, initial):
        def step(_, interval):
            lower, high = interval
            midpoint = (lower + high) / 2
            inside = function(midpoint) < 0
            return (
                jnp.where(inside, midpoint, lower),
                jnp.where(inside, high, midpoint),
            )

        lower, high = jax.lax.fori_loop(
            0, steps, step, (jnp.zeros_like(initial), upper)
        )
        return (lower + high) / 2

    def tangent_solve(linearized, right_hand_side):
        return right_hand_side / linearized(jnp.ones_like(right_hand_side))

    return jax.lax.custom_root(field, upper / 2, solve, tangent_solve)


def _surface(positions, radii, directions, weights, support_directions, config):
    center = positions.mean(axis=0)
    centered = positions - center
    offsets = _soft_support(
        centered, radii, support_directions, config.support_temperature
    )
    directions, weights = _map_directions(centered, radii, directions, weights)
    distance = _radial_roots(
        directions,
        offsets,
        support_directions,
        config.wall_temperature,
        config.root_steps,
    )
    relative = distance[:, None] * directions
    scores = (
        relative @ support_directions.T - offsets
    ) / config.wall_temperature
    gradient = jax.nn.softmax(scores, axis=1) @ support_directions
    normals = gradient / jnp.linalg.norm(gradient, axis=1)[:, None]
    incidence = jnp.sum(normals * directions, axis=1)
    areas = weights * distance**2 / incidence
    return Surface(center + relative, areas, normals)


class Cavity:
    """A fixed-resolution, JIT-compiled smooth convex cavity evaluator."""

    def __init__(self, config: CavityConfig):
        _validate_x64()
        if not isinstance(config, CavityConfig):
            raise TypeError("config must be a CavityConfig")
        self.config = config
        self.reference: ReferenceMesh = mesh_for_target(config.target_points)
        self.shape_reference: ReferenceMesh = mesh_for_target(
            config.target_shape_directions
        )
        directions = jnp.asarray(self.reference.directions)
        weights = jnp.asarray(self.reference.weights)
        support_directions = jnp.asarray(self.shape_reference.directions)
        self._support_directions = support_directions

        def function(positions, radii):
            return _surface(
                positions, radii, directions, weights, support_directions, config
            )

        self.function = jax.jit(function)

        def response(positions, radii, cotangents):
            _, pullback = jax.vjp(lambda p: function(p, radii), positions)
            surface_cotangent = Surface(
                cotangents.points, cotangents.areas, cotangents.normals
            )
            return pullback(surface_cotangent)[0]

        self._response = jax.jit(response)

    @property
    def n_points(self) -> int:
        return len(self.reference.directions)

    @property
    def n_shape_directions(self) -> int:
        return len(self.shape_reference.directions)

    def build(self, positions, radii) -> Surface:
        """Construct points, areas, and outward normals from Bohr inputs."""
        positions_array, radii_array = _validate_atoms(positions, radii)
        return self.function(jnp.asarray(positions_array), jnp.asarray(radii_array))

    def response(self, positions, radii, cotangents: SurfaceCotangent) -> jax.Array:
        """Contract surface cotangents into the nuclear geometry response."""
        positions_array, radii_array = _validate_atoms(positions, radii)
        if not isinstance(cotangents, SurfaceCotangent):
            raise TypeError("cotangents must be a SurfaceCotangent")
        expected = ((self.n_points, 3), (self.n_points,), (self.n_points, 3))
        actual = tuple(np.shape(x) for x in cotangents)
        if actual != expected:
            raise ValueError(f"cotangent shapes must be {expected}, got {actual}")
        if not all(np.all(np.isfinite(np.asarray(x))) for x in cotangents):
            raise ValueError("cotangents must be finite")
        return self._response(
            jnp.asarray(positions_array), jnp.asarray(radii_array),
            SurfaceCotangent(*(jnp.asarray(x) for x in cotangents)),
        )

    def residual(self, positions, radii, surface: Surface) -> jax.Array:
        """Evaluate the implicit-field residual at surface points."""
        positions_array, radii_array = _validate_atoms(positions, radii)
        positions_jax = jnp.asarray(positions_array)
        radii_jax = jnp.asarray(radii_array)
        center = positions_jax.mean(axis=0)
        offsets = _soft_support(
            positions_jax - center,
            radii_jax,
            self._support_directions,
            self.config.support_temperature,
        )
        return _implicit_field(
            surface.points - center,
            offsets,
            self._support_directions,
            self.config.wall_temperature,
        )


def validate_surface(surface: Surface, residual=None, *, tolerance=1e-9) -> None:
    """Validate finite output, positive areas, unit normals, and residual."""
    points, areas, normals = (np.asarray(value) for value in surface)
    if not all(np.all(np.isfinite(value)) for value in (points, areas, normals)):
        raise FloatingPointError("surface contains nonfinite values")
    if np.any(areas <= 0):
        raise FloatingPointError("surface contains nonpositive areas")
    if not np.allclose(np.linalg.norm(normals, axis=1), 1, atol=1e-10):
        raise FloatingPointError("surface normals are not unit length")
    if residual is not None and np.max(np.abs(np.asarray(residual))) > tolerance:
        raise FloatingPointError(
            f"implicit projection residual exceeds {tolerance}: "
            f"{np.max(np.abs(np.asarray(residual)))}"
        )
