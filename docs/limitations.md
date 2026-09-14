# Limitations

CavJAX currently implements a specific smooth convex envelope, not a universal
molecular-surface model.

- **Strict global convexity:** broad voids, ring interiors, and intermolecular
  crevices are filled. This is intentional for enclosing pressure cavities but
  unsuitable where concavity is required.
- **Finite-direction shape:** the implicit envelope is defined by a finite fixed
  direction grid. Increasing only final surface points does not remove its
  orientation dependence.
- **Inflation:** support and wall smoothing enlarge the body. Even a single
  input ball is not reproduced exactly at finite shape resolution and wall
  temperature.
- **Adapted sampling is not equal-area:** the covariance map improves some
  aspect ratios but can stretch patches and does not guarantee uniform Gaussian
  overlap.
- **Fixed topology:** point counts are selected at setup. CavJAX deliberately
  avoids geometry-dependent insertion, deletion, or remeshing.
- **Float64 required:** gradient and root-accuracy guarantees have not been
  established for float32.
- **Limited physical radius policy:** the dependency-free `Cavity` API requires
  callers to provide and justify positive radii. The optional `MolecularCavity`
  convenience uses only the installed PySCF modified-Bondi table, including
  PySCF's generic fallback entries; it does not select among radius models or
  establish that those radii are suitable for every application.

These limitations are model properties, not hidden implementation details.
Applications should test orientation sensitivity, point and shape convergence,
and downstream energy/gradient behavior for their systems.
