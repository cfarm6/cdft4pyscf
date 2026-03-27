# cdft4pyscf

[License](https://github.com/cfarm6/cdft4pyscf/blob/master/LICENSE)
[Powered by: Pixi](https://pixi.sh)
[Code style: ruff](https://github.com/astral-sh/ruff)
[Typing: ty](https://github.com/astral-sh/ty)
[GitHub Workflow Status](https://github.com/cfarm6/cdft4pyscf/actions/)
[Codecov](https://codecov.io/gh/cfarm6/cdft4pyscf)

## Description

`cdft4pyscf` provides constrained density functional theory (cDFT) workflows on top of PySCF and GPU4PySCF.
v0.1 focuses on `dft.UKS` and supports typed Python APIs for:

- atom-list regions
- electron-number constraints (including non-integer targets)
- net-charge constraints
- direct optimization of constrained potentials (`Vc`) integrated into the mean-field `kernel()` cycle

## v0.1 Support Matrix

- **Method:** `dft.UKS` only
- **Backends:** CPU (`pyscf`) and GPU (`gpu4pyscf`) routes
- **Population analysis:** orthogonal-AO populations via PySCF `pyscf.lo.orth_ao` (default: `lowdin`; also supports `meta_lowdin` and `iao`)
- **Constraint composition:** multiple simultaneous constraints, each with its own Lagrange multiplier
- **Interface:** class-first Python API (`CDFT_UKS` mean-field objects)

## Usage

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

## Installation

```bash
pip install -e .
```

GPU support is optional and can be installed via extras:

```bash
pip install -e ".[gpu]"
```

Pixi also provides optional GPU environments:

```bash
pixi shell -e gpu
pixi shell -e dev-gpu
```

## Documentation

The project docs are configured with Zensical.

- Serve locally: `pixi run -e docs docs-serve`
- Build static site: `pixi run -e docs docs-build`

The source pages live under `docs/`, and configuration is in `zensical.toml`.

## Example Script

A more complete example in the style of the PySCF advanced cDFT example is
available at:

- `examples/033_constrained_dft.py`

## Algorithm Notes

Löwdin population for constrained region `C`:

```math
N_c = \mathrm{Tr}(\mathbf{P}\mathbf{w_c^L})
```

with `w_c^L` built from `S^{1/2}` and atom-region AO projector terms.

`Vc` is solved with direct optimization (Wu and Van Voorhis, 2006, Section 2.2)
inside the cDFT mean-field object:

1. Build base Fock inside the SCF cycle.
2. Optimize `Vc` against current base Fock (SciPy root solver).
3. Inject constraint operator into the mean-field Fock construction.
4. Let PySCF SCF machinery (including built-in DIIS controls) continue the cycle.

### Inner `Vc` solve robustness

To reduce late-iteration divergence (for example, residual spikes after apparent
early convergence), the inner multiplier update uses guarded acceptance:

1. Solve for a candidate `Vc` with SciPy `optimize.root(..., method="hybr")`.
2. Compute a bounded step from the previous multiplier:
   `delta = clip(Vc_candidate - Vc_prev, -vc_max_step, vc_max_step)`.
3. Backtrack over damped trial steps (`1.0, 0.5, 0.25, 0.1`) and evaluate the
   residual norm for each.
4. Accept the best residual-improving trial; if the SciPy solve reports failure
   and no trial improves the residual norm, keep `Vc_prev`.

This keeps the `Vc` trajectory stable when the SCF map is non-smooth (for
example, near occupation changes) and avoids committing large unbounded
multiplier jumps from failed inner solves.

## Convergence and Failure Behavior

- SCF and cDFT tolerances are configurable on the mean-field object.
- `vc_max_step` controls the maximum per-constraint `Vc` step accepted each
  inner update (`0.25` by default).
- Fractional electron targets are treated as continuous target values during `Vc` root solving.
- If convergence fails within iteration limits, the class raises `ConvergenceError`.

## Known v0.1 Caveats

- `RKS`/`ROKS`/HF-style references are not yet supported.
- GPU behavior depends on local CUDA/GPU4PySCF availability.
- The current API is Python-only (no CLI in v0.1).

## References

[1] Q. Wu and T. Van Voorhis, “Constrained Density Functional Theory and Its Application in Long-Range Electron Transfer,” J. Chem. Theory Comput., vol. 2, no. 3, pp. 765-774, May 2006, doi: 10.1021/ct0503163.

## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [jevandezande/pixi-cookiecutter](https://github.com/jevandezande/pixi-cookiecutter) project template.
