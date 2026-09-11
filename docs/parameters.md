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

## Inputs deliberately not provided

CavJAX does not choose atomic radii and has no `radius_scale`, `radius_offset`,
`points_per_atom`, atom type, or element table. Applications should compute
explicit `radii_bohr` and translate any per-atom convenience budget into
`target_points` before constructing `CavityConfig`.

There is no public method, quadrature-type, face-centered, or subdivision-level
switch. CavJAX exposes only the selected adapted, vertex-centered path.
