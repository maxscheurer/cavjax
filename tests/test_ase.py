"""Development-only interoperability check; ASE is not a runtime dependency."""

import jax.numpy as jnp
import numpy as np
from ase.build import molecule
from ase.data.vdw import vdw_radii
from ase.units import Bohr

from cavjax import Cavity, CavityConfig, validate_surface


def test_c60_ase_arrays_are_accepted_without_adapter():
    atoms = molecule("C60")
    positions_bohr = jnp.asarray(atoms.positions / Bohr)
    radii_bohr = jnp.asarray(vdw_radii[atoms.numbers] / Bohr)
    cavity = Cavity(CavityConfig(target_points=162, target_shape_directions=42))
    surface = cavity.build(positions_bohr, radii_bohr)
    validate_surface(surface, cavity.residual(positions_bohr, radii_bohr, surface))
    assert len(surface.points) == cavity.n_points
    assert np.all(np.asarray(surface.areas) > 0)
