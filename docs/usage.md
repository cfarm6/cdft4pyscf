# Usage

## Core API

The primary entry point is the constrained mean-field class:

- `CDFT_UKS(mol, *, constraints, ...)`

You can also use `build_cdft_mean_field(request: RunRequest)` to construct a
configured backend-specific cDFT mean-field object from typed request models.

## Minimal workflow

1. Define one or more `ConstraintSpec` values.
2. For `electron_number` constraints, set `region` to a `RegionSpec` or
   `list[RegionSpec]`.
3. For `net_charge` constraints, set `region` to a single `RegionSpec`.
4. Build a PySCF molecule (`gto.M`).
5. Construct `CDFT_UKS` and set standard mean-field settings (e.g. `xc`).
6. Call `mf.kernel()` and inspect cDFT helpers (`constraint_values`, `multiplier_by_constraint`).

## Example

```python
from pyscf import gto

from cdft4pyscf import CDFT_UKS, ConstraintSpec, RegionSpec

mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
mf = CDFT_UKS(
    mol,
    constraints=[
        ConstraintSpec(
            name="n_frag_a",
            kind="electron_number",
            target=1.0,
            region=RegionSpec(name="frag_a", atom_indices=[0]),
        ),
        ConstraintSpec(
            name="q_total",
            kind="net_charge",
            target=0.0,
            region=RegionSpec(name="all_atoms", atom_indices=[0, 1]),
        ),
    ],
)
mf.xc = "lda,vwn"
energy = mf.kernel()
print(energy)
print(mf.constraint_values())
print(mf.multiplier_by_constraint())
```

## Solver behavior

- `CDFT_UKS` performs constrained SCF updates and optimizes `Vc` for each
  constraint during `kernel()`.
- Fractional electron targets are supported.
- If convergence fails within iteration limits, `ConvergenceError` is raised.
