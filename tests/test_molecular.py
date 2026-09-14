import os
import subprocess
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from cavjax import Cavity, MolecularCavity, SurfaceCotangent


POSITIONS = jnp.array([
    [-2.0, 0.0, 0.2],
    [1.0, 2.0, -0.1],
    [2.0, -1.0, 0.4],
])
NUMBERS = np.array([1, 6, 8])


def test_import_and_low_level_api_do_not_require_pyscf():
    code = r"""
import builtins
import sys

import jax
jax.config.update("jax_enable_x64", True)

original_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "pyscf" or name.startswith("pyscf."):
        raise ModuleNotFoundError("blocked PySCF for optional-dependency test")
    return original_import(name, *args, **kwargs)
builtins.__import__ = blocked_import

import cavjax
assert not any(name == "pyscf" or name.startswith("pyscf.") for name in sys.modules)
cavity = cavjax.Cavity(cavjax.CavityConfig(target_points=12))
assert cavity.n_points == 12
try:
    cavjax.MolecularCavity([1], points_per_atom=12)
except ImportError as error:
    assert "optional PySCF dependency" in str(error)
    assert "cavjax[pyscf]" in str(error)
else:
    raise AssertionError("MolecularCavity should require PySCF")
"""
    environment = dict(os.environ, JAX_ENABLE_X64="1")
    subprocess.run([sys.executable, "-c", code], env=environment, check=True)


def test_radius_formula_provenance_and_fixed_state(monkeypatch):
    from pyscf.data import radii
    from pyscf.solvent import pcm

    fallback = np.flatnonzero(
        np.isclose(pcm.modified_Bondi, 1.999999 / radii.BOHR)
    )
    fallback = fallback[fallback > 0][0]
    atomic_numbers = np.array([1, 6, fallback])
    original_numbers = atomic_numbers.copy()
    scale = 1.2
    offset = 0.15
    expected = scale * pcm.modified_Bondi[original_numbers] + offset

    cavity = MolecularCavity(
        atomic_numbers,
        points_per_atom=14,
        radius_scale=scale,
        radius_offset=offset,
        target_shape_directions=42,
    )

    np.testing.assert_array_equal(cavity.atomic_numbers, original_numbers)
    np.testing.assert_allclose(cavity.effective_radii, expected)

    atomic_numbers[:] = 8
    changed_table = pcm.modified_Bondi.copy()
    changed_table[original_numbers] *= 3
    monkeypatch.setattr(pcm, "modified_Bondi", changed_table)
    exposed_numbers = cavity.atomic_numbers
    exposed_radii = cavity.effective_radii
    exposed_numbers[:] = 2
    exposed_radii[:] = 100

    np.testing.assert_array_equal(cavity.atomic_numbers, original_numbers)
    np.testing.assert_allclose(cavity.effective_radii, expected)


@pytest.mark.parametrize(
    "atomic_numbers",
    [
        [],
        [[1, 6]],
        [1.0, 6.0],
        [True, False],
        [0],
        [-1],
        [104],
    ],
)
def test_atomic_number_validation(atomic_numbers):
    with pytest.raises(ValueError):
        MolecularCavity(atomic_numbers, points_per_atom=12)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("points_per_atom", 0),
        ("points_per_atom", -1),
        ("points_per_atom", np.nan),
        ("points_per_atom", np.inf),
        ("points_per_atom", True),
        ("radius_scale", 0),
        ("radius_scale", -1),
        ("radius_scale", np.nan),
        ("radius_scale", np.inf),
        ("radius_offset", np.nan),
        ("radius_offset", np.inf),
    ],
)
def test_molecular_option_validation(keyword, value):
    with pytest.raises(ValueError):
        MolecularCavity([1], **{keyword: value})


def test_target_overflow_and_invalid_effective_radii():
    with pytest.raises(ValueError, match="target_points"):
        MolecularCavity([1, 6], points_per_atom=np.finfo(float).max)
    with pytest.raises(ValueError, match="effective radii"):
        MolecularCavity([1], radius_offset=-100.0)


@pytest.mark.parametrize("bad_value", [0.0, np.nan, np.inf])
def test_invalid_source_radius_is_rejected(monkeypatch, bad_value):
    from pyscf.solvent import pcm

    table = pcm.modified_Bondi.copy()
    table[1] = bad_value
    monkeypatch.setattr(pcm, "modified_Bondi", table)
    with pytest.raises(ValueError, match="modified_Bondi radii"):
        MolecularCavity([1])


def test_malformed_source_table_is_rejected(monkeypatch):
    from pyscf.solvent import pcm

    monkeypatch.setattr(pcm, "modified_Bondi", np.ones((2, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        MolecularCavity([1])


def test_point_budget_fixed_topology_and_low_level_equivalence():
    molecular = MolecularCavity(
        NUMBERS,
        points_per_atom=14,
        radius_scale=1.1,
        radius_offset=0.05,
        target_shape_directions=42,
        support_temperature=0.25,
        wall_temperature=0.15,
        root_steps=48,
    )
    assert molecular.config.target_points == 42
    assert molecular.config.root_steps == 48
    assert molecular.n_points == 42
    assert molecular.n_shape_directions == 42

    low_level = Cavity(molecular.config)
    expected = low_level.build(POSITIONS, molecular.effective_radii)
    actual = molecular.build(POSITIONS)
    for actual_value, expected_value in zip(actual, expected):
        np.testing.assert_allclose(actual_value, expected_value, atol=1e-12)

    moved = molecular.build(POSITIONS + jnp.array([0.2, -0.3, 0.4]))
    assert len(moved.points) == molecular.n_points == len(actual.points)

    cotangents = SurfaceCotangent(
        points=jnp.arange(actual.points.size, dtype=float).reshape(actual.points.shape) / 100,
        areas=jnp.linspace(-0.5, 0.5, molecular.n_points),
        normals=jnp.ones_like(actual.normals) * 0.2,
    )
    expected_response = low_level.response(
        POSITIONS, molecular.effective_radii, cotangents
    )
    actual_response = molecular.response(POSITIONS, cotangents)
    np.testing.assert_allclose(actual_response, expected_response, atol=1e-12)

    with pytest.raises(ValueError):
        molecular.build(POSITIONS[:2])
    with pytest.raises(ValueError):
        molecular.response(POSITIONS[:2], cotangents)
