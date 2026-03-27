#!/usr/bin/env python
"""Water cDFT example with wrapper-first API."""

from __future__ import annotations

from pyscf import dft, gto

from cdft4pyscf import CDFT, Constraint, FragmentTerm, ProjectorSpec, SolverOptions
from cdft4pyscf.exceptions import ConvergenceError


def main() -> None:
    """Run a small constrained DFT calculation."""
    atom = """O    0    0    0
        H    0.   -0.757   0.587
        H    0.   0.757    0.587"""

    mol = gto.M(atom=atom, basis="6-31g", charge=0, spin=0, verbose=3)
    base = dft.UKS(mol)
    base.xc = "b3lyp"
    base.max_cycle = 80

    mf = CDFT(
        base,
        constraints=[
            Constraint(
                name="n_oxygen_fragment",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=8.2,
                target_type="electrons",
                projector=ProjectorSpec(method="iao"),
            ),
            Constraint(
                name="n_hydrogen_fragment",
                fragments=[FragmentTerm(atoms=[1], coeff=1.0)],
                target=0.6,
                target_type="electrons",
                projector=ProjectorSpec(method="iao"),
            ),
        ],
        solver=SolverOptions(
            mode="micro",
            initial_v_lagrange=[0.25, 0.001],
            conv_tol_constraint=1e-7,
            inner_vc_tol=1e-7,
            inner_vc_max_cycle=80,
            trace=True,
        ),
    )

    try:
        total_energy = mf.kernel()
    except ConvergenceError as exc:
        print(f"cDFT did not converge: {exc}")
        return

    print("Converged:", mf.converged)
    print("Total energy (Eh):", total_energy)
    print("Constraint values:", mf.constraint_values())
    print("Lagrange multipliers (v_lagrange):", mf.v_lagrange)
    print("Constraint residuals:", mf.constraint_residuals())


if __name__ == "__main__":
    main()
