#!/usr/bin/env python
"""Donor-acceptor difference-constraint cDFT example."""

from __future__ import annotations

from gpu4pyscf.dft import UKS
from pyscf import gto

from cdft4pyscf import CDFT, Constraint, FragmentTerm, ProjectorSpec, SolverOptions
from cdft4pyscf.exceptions import ConvergenceError


def main() -> None:
    """Run a simple electron-difference constraint on a dimer."""
    atom = """
    C  0.000  0.000  0.000
    O  0.000  0.000  1.200
    C  0.000  0.000  3.200
    O  0.000  0.000  4.400
    """
    mol = gto.M(atom=atom, basis="6-31g", charge=0, spin=0, verbose=4)
    base = UKS(mol)
    base.xc = "b3lyp"
    base.max_cycle = 100

    mf = CDFT(
        base,
        constraints=[
            Constraint(
                name="delta_N_donor_acceptor",
                fragments=[
                    FragmentTerm(atoms=[0, 1], coeff=1.0),
                    FragmentTerm(atoms=[2, 3], coeff=-1.0),
                ],
                target=0.25,
                target_type="electrons",
                projector=ProjectorSpec(method="mulliken"),
            )
        ],
        solver=SolverOptions(
            mode="micro",
            trace=True,
            conv_tol_constraint=1e-6,
            max_v_step=0.5,
            inner_vc_max_cycle=100,
            verbosity=4,
        ),
    )

    try:
        energy = mf.kernel()
    except ConvergenceError as exc:
        print(f"cDFT did not converge: {exc}")
        return

    print("Converged:", mf.converged)
    print("Total energy (Eh):", energy)
    print("Constraint values:", mf.constraint_values())
    print("Constraint residuals:", mf.constraint_residuals())
    print("v_lagrange:", mf.v_lagrange)


if __name__ == "__main__":
    main()
