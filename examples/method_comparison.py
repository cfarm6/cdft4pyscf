#!/usr/bin/env python
"""Projector backend comparison example for one fragment constraint."""

from __future__ import annotations

from typing import Literal

from gpu4pyscf import dft
from pyscf import gto

from cdft4pyscf import CDFT, Constraint, FragmentTerm, ProjectorSpec, SolverOptions
from cdft4pyscf.exceptions import ConvergenceError

ProjectorMethod = Literal["mulliken", "lowdin", "minao", "iao", "becke"]


def run_with_method(method: ProjectorMethod) -> tuple[float, dict[str, float], list[float]]:
    """Run one cDFT job and return energy, values, and multipliers."""
    mol = gto.M(
        atom="O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
        basis="def2-svp",
        charge=0,
        spin=0,
        verbose=0,
    )
    base = dft.UKS(mol)
    base.xc = "b3lyp"
    base.max_cycle = 60
    mf = CDFT(
        base,
        constraints=[
            Constraint(
                name="oxygen_population",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=8.1,
                target_type="electrons",
                projector=ProjectorSpec(method=method),
            )
        ],
        solver=SolverOptions(mode="micro", conv_tol_constraint=1e-6),
    )
    energy = mf.kernel()
    return energy, mf.constraint_values(), mf.v_lagrange.tolist()


def main() -> None:
    """Compare projector effects for the same target constraint."""
    methods: list[ProjectorMethod] = ["mulliken", "lowdin", "minao", "becke"]
    for method in methods:
        try:
            energy, values, multipliers = run_with_method(method)
        except ConvergenceError as exc:
            print(f"{method}: unconverged ({exc})")
            continue
        print(f"{method}:")
        print(f"  energy: {energy:.10f}")
        print(f"  values: {values}")
        print(f"  v_lagrange: {multipliers}")


if __name__ == "__main__":
    main()
