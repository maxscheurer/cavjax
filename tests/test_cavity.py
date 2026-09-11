from dataclasses import replace
import os
import subprocess
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from cavjax import Cavity, CavityConfig, SurfaceCotangent, validate_surface


POSITIONS = jnp.array([[-2.0, 0.0, 0.2], [1.0, 2.0, -0.1], [2.0, -1.0, 0.4]])
RADII = jnp.array([1.4, 1.8, 1.2])
CONFIG = CavityConfig(
    target_points=42,
    target_shape_directions=42,
    support_temperature=0.25,
    wall_temperature=0.15,
)


def test_surface_validity_enclosure_translation_and_permutation():
    cavity = Cavity(CONFIG)
    surface = cavity.build(POSITIONS, RADII)
    validate_surface(surface, cavity.residual(POSITIONS, RADII, surface))

    gaps = (
        np.linalg.norm(np.asarray(surface.points)[:, None] - np.asarray(POSITIONS), axis=2)
        - np.asarray(RADII)
    )
    assert gaps.min() >= -1e-10

    shift = jnp.array([0.5, -0.3, 1.2])
    shifted = cavity.build(POSITIONS + shift, RADII)
    np.testing.assert_allclose(shifted.points, surface.points + shift, atol=1e-11)
    np.testing.assert_allclose(shifted.areas, surface.areas, atol=1e-11)
    np.testing.assert_allclose(shifted.normals, surface.normals, atol=1e-11)

    permutation = jnp.array([2, 0, 1])
    permuted = cavity.build(POSITIONS[permutation], RADII[permutation])
    for actual, expected in zip(permuted, surface):
        np.testing.assert_allclose(actual, expected, atol=1e-11)


def test_each_output_directional_derivative_and_vjp_duality():
    cavity = Cavity(CONFIG)
    rng = np.random.default_rng(41)
    direction = jnp.asarray(rng.normal(size=POSITIONS.shape))
    surface, tangent = jax.jvp(
        lambda positions: cavity.function(positions, RADII),
        (POSITIONS,),
        (direction,),
    )
    step = 1e-4
    plus = cavity.function(POSITIONS + step * direction, RADII)
    minus = cavity.function(POSITIONS - step * direction, RADII)
    for automatic, upper, lower in zip(tangent, plus, minus):
        np.testing.assert_allclose(
            automatic, (upper - lower) / (2 * step), rtol=2e-5, atol=3e-7
        )

    cotangents = SurfaceCotangent(
        *(jnp.asarray(rng.normal(size=value.shape)) for value in surface)
    )
    response = cavity.response(POSITIONS, RADII, cotangents)
    forward_contraction = sum(
        jnp.vdot(value, cotangent)
        for value, cotangent in zip(tangent, cotangents)
    )
    np.testing.assert_allclose(
        forward_contraction, jnp.vdot(response, direction), atol=2e-9
    )
    assert np.linalg.norm(np.asarray(tangent.normals)) > 1e-3


def test_bohr_dimensional_scaling():
    scale = 1.7
    base = Cavity(CONFIG)
    scaled_config = replace(
        CONFIG,
        support_temperature=CONFIG.support_temperature * scale,
        wall_temperature=CONFIG.wall_temperature * scale,
    )
    scaled = Cavity(scaled_config)
    surface = base.build(POSITIONS, RADII)
    transformed = scaled.build(scale * POSITIONS, scale * RADII)
    np.testing.assert_allclose(transformed.points, scale * surface.points, atol=2e-10)
    np.testing.assert_allclose(transformed.areas, scale**2 * surface.areas, atol=2e-10)
    np.testing.assert_allclose(transformed.normals, surface.normals, atol=2e-10)


def test_import_does_not_enable_x64_and_construction_requires_it():
    code = """
import jax
assert not jax.config.jax_enable_x64
import cavjax
assert not jax.config.jax_enable_x64
try:
    cavjax.Cavity(cavjax.CavityConfig(target_points=12))
except RuntimeError as error:
    assert 'requires JAX float64' in str(error)
else:
    raise AssertionError('Cavity construction should reject disabled float64')
"""
    environment = dict(os.environ, JAX_ENABLE_X64="0")
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def test_input_and_cotangent_validation():
    cavity = Cavity(CONFIG)
    with pytest.raises(ValueError):
        cavity.build([], [])
    with pytest.raises(ValueError):
        cavity.build([[0.0, 0.0, 0.0]], [-1.0])
    with pytest.raises(TypeError):
        cavity.response(POSITIONS, RADII, None)
    bad = SurfaceCotangent(jnp.zeros((1, 3)), jnp.zeros(1), jnp.zeros((1, 3)))
    with pytest.raises(ValueError):
        cavity.response(POSITIONS, RADII, bad)
