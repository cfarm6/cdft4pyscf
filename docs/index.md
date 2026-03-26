# cdft4pyscf

`cdft4pyscf` provides constrained density functional theory (cDFT) workflows on
top of PySCF and GPU4PySCF.

## What v0.1 supports

- `dft.UKS` workflows
- CPU (`pyscf`) and GPU (`gpu4pyscf`) backends
- Löwdin population analysis for constrained regions
- Multiple simultaneous constraints with independent Lagrange multipliers

## Quick start

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
print(mf.converged, energy, mf.multiplier_by_constraint())
```

## Where to go next

- [Install](installation.md) for Pixi and `pip` setup
- [Quickstart](usage.md) for a first constrained run
- [Examples](examples.md) for end-to-end scripts
- [API Overview](api.md) for API entry points
- [API by Workflow](api-workflow.md) for task-oriented API navigation
- [API Reference](api-reference.md) for complete symbol details
