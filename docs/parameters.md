# Parameters and units

CavJAX uses **Bohr** for every length and **Bohr²** for every area. It has no
unit switch. Radii, coordinates, and smoothing temperatures must use the same
Bohr convention.

## `CavityConfig`

| Parameter | Default | Meaning | Cost/effect |
|---|---:|---|---|
| `target_points` | required | Approximate global final surface-point budget | Dominant integration and VJP cost |
| `target_shape_directions` | 162 | Approximate number of directions defining the implicit envelope | Controls shape resolution and orientation error; increases field cost |
| `support_temperature` | 0.2834589187 Bohr | Smooth atomic support temperature | Larger values make the support envelope smoother and more inflated |
| `wall_temperature` | 0.1511780900 Bohr | Smooth halfspace-envelope temperature | Larger values smooth and inflate the final implicit surface |
| `root_steps` | 48 | Fixed bisection steps for radial projection | Controls projection accuracy and forward cost; minimum 32 |

The two target counts are rounded to the nearest valid frequency-grid count

```text
Q(f) = 10 f² + 2.
```

The resolved values are available as `cavity.n_points` and
`cavity.n_shape_directions`. Molecular geometry never changes the chosen point
count or topology.

## `MolecularCavity`

`MolecularCavity` is an optional PySCF-backed setup layer. It accepts a nonempty
one-dimensional integer array of positive atomic numbers and stores fixed
modified-Bondi effective radii. Install it with `pip install
"cavjax[pyscf]"`. Importing CavJAX and using the low-level API does not import
or require PySCF.

| Parameter | Default | Meaning |
|---|---:|---|
| `points_per_atom` | 100 | Positive finite convenience multiplier for the global target |
| `radius_scale` | 1.0 | Positive finite multiplier for PySCF modified-Bondi radii |
| `radius_offset` | 0.0 Bohr | Finite offset applied after radius scaling |
| `target_shape_directions` | 162 | Same global shape target as `CavityConfig` |
| `support_temperature` | 0.2834589187 Bohr | Same support smoothing parameter as `CavityConfig` |
| `wall_temperature` | 0.1511780900 Bohr | Same wall smoothing parameter as `CavityConfig` |
| `root_steps` | 48 | Same fixed root-solver iteration count as `CavityConfig` |

At construction, the wrapper reads the installed
`pyscf.solvent.pcm.modified_Bondi` table and computes

```text
effective_radii = radius_scale * modified_Bondi[atomic_numbers] + radius_offset
target_points = len(atomic_numbers) * points_per_atom
```

PySCF stores this table in Bohr, including a hydrogen radius of 1.10 Å converted
to Bohr. CavJAX accepts its finite positive generic fallback entries as-is and
does not embed or substitute a second periodic-table dataset. Atomic number
zero (PySCF's ghost/unknown slot), negative values, booleans, floating arrays,
and indices outside the installed table are rejected. Source and effective
radii must be finite and positive.

`points_per_atom` may be any finite positive scalar because `target_points` is
an approximate target. It selects one **global** grid, not one atom-centered
grid per atom. The target is rounded to `Q(f)`, and the resulting topology is
fixed when the wrapper constructs its retained low-level `Cavity`.

The fixed inputs and resolved setup are inspectable through
`atomic_numbers`, `effective_radii`, `config`, `n_points`, and
`n_shape_directions`. Returned arrays cannot be used to mutate the wrapper's
stored molecular state. New positions passed to `build` or `response` must have
the same atom count.

## Low-level inputs and unsupported switches

`Cavity` continues to accept explicit radii and is the supported path for
custom radius models or installations without PySCF. Applications using it
must compute `radii_bohr` and any global `target_points` policy themselves.

There is no public method, quadrature-type, face-centered, atom-ownership, or
subdivision-level switch. CavJAX exposes only the selected adapted,
vertex-centered path.
