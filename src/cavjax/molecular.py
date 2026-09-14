"""PySCF-backed molecular setup for fixed CavJAX cavities.

All lengths are in Bohr. PySCF is imported only when ``MolecularCavity`` is
constructed; the low-level cavity API does not depend on it.
"""

from __future__ import annotations

import jax
import numpy as np

from .cavity import Cavity, CavityConfig, Surface, SurfaceCotangent


_PYSCF_ERROR = (
    "MolecularCavity requires the optional PySCF dependency. "
    "Install it with `pip install 'cavjax[pyscf]'`."
)


def _finite_scalar(value, name: str, *, positive: bool) -> float:
    """Validate and convert a real scalar option."""
    try:
        array = np.asarray(value)
        if array.shape != () or np.issubdtype(array.dtype, np.bool_):
            raise ValueError
        result = float(array)
    except (TypeError, ValueError, OverflowError) as error:
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}") from error
    if not np.isfinite(result) or (positive and result <= 0):
        qualifier = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _validate_atomic_numbers(atomic_numbers) -> np.ndarray:
    try:
        array = np.asarray(atomic_numbers)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "atomic_numbers must be a nonempty one-dimensional integer array"
        ) from error
    if (
        array.ndim != 1
        or len(array) == 0
        or np.issubdtype(array.dtype, np.bool_)
        or not np.issubdtype(array.dtype, np.integer)
    ):
        raise ValueError(
            "atomic_numbers must be a nonempty one-dimensional integer array"
        )
    result = np.array(array, dtype=np.int64, copy=True)
    if np.any(result <= 0):
        raise ValueError("atomic_numbers must contain positive nuclear charges")
    result.setflags(write=False)
    return result


def _load_modified_bondi() -> np.ndarray:
    try:
        from pyscf.solvent.pcm import modified_Bondi
    except ImportError as error:
        raise ImportError(_PYSCF_ERROR) from error

    table = np.asarray(modified_Bondi)
    if table.ndim != 1:
        raise ValueError("PySCF modified_Bondi must be a one-dimensional table")
    return table


class MolecularCavity:
    """A fixed molecular cavity using PySCF modified-Bondi radii.

    ``atomic_numbers`` and the derived effective radii are fixed at
    construction. ``points_per_atom`` selects one global cavity grid through
    ``len(atomic_numbers) * points_per_atom``; it does not create atom-centered
    grids. All length-valued parameters are in Bohr.
    """

    def __init__(
        self,
        atomic_numbers,
        *,
        points_per_atom: float = 100,
        radius_scale: float = 1.0,
        radius_offset: float = 0.0,
        target_shape_directions: float = 162,
        support_temperature: float = 0.2834589186938655,
        wall_temperature: float = 0.15117808997006163,
        root_steps: int = 48,
    ) -> None:
        numbers = _validate_atomic_numbers(atomic_numbers)
        point_budget = _finite_scalar(
            points_per_atom, "points_per_atom", positive=True
        )
        scale = _finite_scalar(radius_scale, "radius_scale", positive=True)
        offset = _finite_scalar(radius_offset, "radius_offset", positive=False)

        target_points = len(numbers) * point_budget
        if not np.isfinite(target_points) or target_points <= 0:
            raise ValueError("target_points must be finite and positive")

        table = _load_modified_bondi()
        if np.any(numbers >= len(table)):
            raise ValueError("atomic_numbers contain an index outside modified_Bondi")
        try:
            source_radii = np.array(table[numbers], dtype=float, copy=True)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                "modified_Bondi radii must be finite and positive"
            ) from error
        if not np.all(np.isfinite(source_radii)) or np.any(source_radii <= 0):
            raise ValueError("modified_Bondi radii must be finite and positive")

        with np.errstate(over="ignore", invalid="ignore"):
            effective_radii = scale * source_radii + offset
        if not np.all(np.isfinite(effective_radii)) or np.any(effective_radii <= 0):
            raise ValueError("effective radii must be finite and positive")
        effective_radii.setflags(write=False)

        config = CavityConfig(
            target_points=target_points,
            target_shape_directions=target_shape_directions,
            support_temperature=support_temperature,
            wall_temperature=wall_temperature,
            root_steps=root_steps,
        )
        self._atomic_numbers = numbers
        self._effective_radii = effective_radii
        self._cavity = Cavity(config)

    @property
    def atomic_numbers(self) -> np.ndarray:
        """Return a copy of the fixed positive nuclear charges."""
        return self._atomic_numbers.copy()

    @property
    def effective_radii(self) -> np.ndarray:
        """Return a copy of the fixed effective radii in Bohr."""
        return self._effective_radii.copy()

    @property
    def config(self) -> CavityConfig:
        """The frozen low-level cavity configuration."""
        return self._cavity.config

    @property
    def n_points(self) -> int:
        """Actual number of points in the fixed global cavity grid."""
        return self._cavity.n_points

    @property
    def n_shape_directions(self) -> int:
        """Actual number of directions in the fixed shape grid."""
        return self._cavity.n_shape_directions

    def build(self, positions) -> Surface:
        """Construct the surface for Bohr positions using the fixed radii."""
        return self._cavity.build(positions, self._effective_radii)

    def response(
        self, positions, cotangents: SurfaceCotangent
    ) -> jax.Array:
        """Contract surface cotangents into the nuclear geometry response."""
        return self._cavity.response(
            positions, self._effective_radii, cotangents
        )
