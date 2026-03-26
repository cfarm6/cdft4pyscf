#!/usr/bin/env python
"""Constrained DFT example using class-first kernel workflows."""

from __future__ import annotations

from pyscf import gto

from cdft4pyscf import CDFT_UKS, ConstraintSpec, RegionSpec
from cdft4pyscf.exceptions import ConvergenceError


def main() -> None:
    """Run a small constrained DFT calculation."""
    atom = """O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587"""

    mol = gto.M(atom=atom, basis="6-31g", charge=0, spin=0, verbose=3)
    mf = CDFT_UKS(
        mol,
        constraints=[
            ConstraintSpec(
                name="n_oxygen_fragment",
                kind="electron_number",
                target=8.2,
                region=RegionSpec(name="oxygen_fragment", atom_indices=[0]),
            ),
            ConstraintSpec(
                name="n_hydrogen_fragment",
                kind="electron_number",
                target=0.6,
                region=RegionSpec(name="hydrogen_fragment", atom_indices=[1]),
            ),
        ],
        initial_vc=[0.25, 0.001],
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf.xc = "b3lyp"
    mf.max_cycle = 80

    try:
        total_energy = mf.kernel()
    except ConvergenceError as exc:
        print(f"cDFT did not converge: {exc}")
        return

    print("Converged:", mf.converged)
    print("Total energy (Eh):", total_energy)
    print("Constraint values:", mf.constraint_values())
    print("Lagrange multipliers (Vc):", mf.multiplier_by_constraint())
    print("Constraint residuals:", mf.constraint_residuals())


if __name__ == "__main__":
    main()
