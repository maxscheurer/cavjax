# CavJAX

CavJAX constructs differentiable smooth convex molecular cavities with JAX.
It maps Cartesian atom centers and positive radii to surface points, positive
quadrature areas, and outward unit normals, and contracts arbitrary surface
cotangents into nuclear geometry responses with a reverse-mode VJP.

> **Status:** pre-alpha research software. The implementation is clean and
> tested, but the current finite-direction envelope has documented shape
> inflation and orientation dependence. See [limitations](docs/limitations.md).

CavJAX's low-level API is independent of PySCF and molecular file formats. It
accepts explicit position and radius arrays. An optional PySCF-backed
`MolecularCavity` adds modified-Bondi radius lookup and a per-atom point-budget
convenience while remaining array-based.

## Install for development

```bash
git clone git@github.com:maxscheurer/cavjax.git
cd cavjax
uv sync --group test
JAX_ENABLE_X64=1 uv run pytest
```

CavJAX requires JAX float64 but deliberately does not mutate process-global JAX
configuration. Set `JAX_ENABLE_X64=1` before starting Python, or enable x64 with
JAX before constructing a cavity.

PySCF is not a mandatory dependency. Install the molecular convenience API with

```bash
pip install "cavjax[pyscf]"
```

Importing `cavjax` and using `Cavity` or `CavityConfig` never imports PySCF.

## Low-level explicit-radius use

```python
import jax.numpy as jnp
from cavjax import Cavity, CavityConfig, SurfaceCotangent

positions = jnp.array([
    [-2.0, 0.0, 0.2],
    [ 1.0, 2.0, -0.1],
    [ 2.0, -1.0, 0.4],
])  # Bohr
radii = jnp.array([1.4, 1.8, 1.2])  # Bohr

cavity = Cavity(CavityConfig(target_points=1000))
surface = cavity.build(positions, radii)

cotangents = SurfaceCotangent(
    points=jnp.zeros_like(surface.points),
    areas=jnp.ones_like(surface.areas),
    normals=jnp.zeros_like(surface.normals),
)
d_area_d_nuclei = cavity.response(positions, radii, cotangents)
```

`surface.points` is in Bohr, `surface.areas` in Bohr², and
`surface.normals` points outward. `response` returns shape `(N, 3)` and does not
materialize a dense surface-by-nucleus Jacobian.

## PySCF-backed molecular use

```python
import jax.numpy as jnp
from cavjax import MolecularCavity, SurfaceCotangent

atomic_numbers = jnp.array([8, 1, 1])
positions_bohr = jnp.array([
    [0.0, 0.0, 0.0],
    [1.43, 1.11, 0.0],
    [-1.43, 1.11, 0.0],
])

cavity = MolecularCavity(
    atomic_numbers,
    points_per_atom=100,
    radius_scale=1.0,
    radius_offset=0.0,  # Bohr
)
surface = cavity.build(positions_bohr)

cotangents = SurfaceCotangent(
    points=jnp.zeros_like(surface.points),
    areas=jnp.ones_like(surface.areas),
    normals=jnp.zeros_like(surface.normals),
)
d_area_d_nuclei = cavity.response(positions_bohr, cotangents)
```

`MolecularCavity` reads the installed
`pyscf.solvent.pcm.modified_Bondi` table once at construction and fixes

```text
effective_radii = radius_scale * modified_Bondi[atomic_numbers] + radius_offset
target_points = len(atomic_numbers) * points_per_atom
```

The table and both formulae use Bohr. `points_per_atom` selects one global,
fixed cavity grid; it is not an atom-centered grid. The requested global target
is rounded to a supported icosahedral count, available as `cavity.n_points`.
The effective radii, configuration, and shape count are available as
`cavity.effective_radii`, `cavity.config`, and
`cavity.n_shape_directions`. PySCF's positive generic fallback radii are used
as provided rather than replaced by a second CavJAX table.

## Scope

CavJAX intentionally provides one cavity algorithm and one vertex-centered
quadrature path. The optional molecular wrapper provides only PySCF's
modified-Bondi radius model; CavJAX does not provide additional radius models,
pressure models, electronic structure, molecular-object adapters, file readers,
visualization, or rejected workshop samplers.

- [Theory](docs/theory.md)
- [Parameters](docs/parameters.md)
- [Limitations](docs/limitations.md)

## License

Apache-2.0.
