# CavJAX

CavJAX constructs differentiable smooth convex molecular cavities with JAX.
It maps Cartesian atom centers and positive radii to surface points, positive
quadrature areas, and outward unit normals, and contracts arbitrary surface
cotangents into nuclear geometry responses with a reverse-mode VJP.

> **Status:** pre-alpha research software. The implementation is clean and
> tested, but the current finite-direction envelope has documented shape
> inflation and orientation dependence. See [limitations](docs/limitations.md).

CavJAX is independent of PySCF and molecular file formats. It accepts arrays;
applications own atom typing, radius models, unit conversion, and downstream
energy expressions.

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

## Minimal use

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

## Scope

CavJAX intentionally provides one cavity algorithm and one vertex-centered
quadrature path. It does not provide atomic radii, pressure models, electronic
structure, file readers, visualization, or rejected workshop samplers.

- [Theory](docs/theory.md)
- [Parameters](docs/parameters.md)
- [Limitations](docs/limitations.md)

## License

Apache-2.0.
