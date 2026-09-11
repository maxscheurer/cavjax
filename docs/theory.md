# Mathematical definition

For atom centers `R_i`, positive radii `r_i`, and a fixed antipodally spanning
set of unit directions `v_m`, CavJAX works relative to the mean center `c`.

The smoothed support offsets are

```text
h_m = τ log Σ_i exp((v_m · (R_i-c) + r_i) / τ).
```

The convex implicit field is

```text
F(y) = η [log Σ_m exp((v_m · y - h_m) / η) - log M].
```

The cavity is the zero level set `F(y)=0`. Every complete input ball lies in the
`F<=0` sublevel set. Fixed antipodally spanning directions make it bounded.

## Adapted reference map

A smooth affine map is built from centered nuclear covariance plus the square
of the mean radius. Fixed Newton–Schulz iterations avoid eigenvector derivatives.
Each reference direction is mapped to a ray `u`; radial distance `ρ` solves
`F(ρu)=0`. Surface quantities are

```text
x = c + ρu
n = ∇F / |∇F|
A = w J_angular ρ² / (n · u).
```

The quadrature uses fixed positive vertex dual-cell solid-angle weights. No
geometry-dependent triangulation, pruning, atom ownership, or remeshing occurs.

## Differentiation

Radial projection uses `jax.lax.custom_root`, so derivatives follow the
converged scalar root equation rather than differentiating bisection decisions.
`Cavity.response` forms a JAX VJP of points, areas, and normals and contracts it
with caller-provided cotangents. The output is only `(N_atoms, 3)`; no dense
surface-by-nucleus Jacobian is constructed.
