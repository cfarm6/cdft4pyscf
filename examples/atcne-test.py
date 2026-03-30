#!/usr/bin/env python
"""ATCNE cDFT smoke test with wrapper-first API."""

from __future__ import annotations

from pathlib import Path

from gpu4pyscf.dft import UKS
from pyscf import gto

from cdft4pyscf import CDFT, Constraint, FragmentTerm, ProjectorSpec, SolverOptions

working_dir = Path(__file__).parent
chk_dir = working_dir / "chkfiles"
chk_dir.mkdir(exist_ok=True, parents=True)
structure_dir = working_dir / "structures"
structure_file = next(structure_dir.glob("*.xyz"))

mol = gto.M(atom=structure_file.as_posix(), basis="def2-svp", charge=0, spin=0, verbose=4)
base = UKS(mol)
base.xc = "b3lyp-d3bj"
base.max_cycle = 100
base.chkfile = chk_dir / "atcne.chk"
base.init_guess = "chkfile"
energy = base.kernel()

# Now run cDFT
mf = CDFT(
    base,
    constraints=[
        Constraint(
            name="anthracene_charge",
            fragments=[FragmentTerm(atoms=list(range(24)), coeff=1.0)],
            target=0.0,
            target_type="charge",
            projector=ProjectorSpec(method="iao"),
        )
    ],
    solver=SolverOptions(mode="newton_kkt", conv_tol_constraint=1e-6),
)
e0 = mf.kernel()

# Now run cDFT
mf = CDFT(
    base,
    constraints=[
        Constraint(
            name="anthracene_charge",
            fragments=[FragmentTerm(atoms=list(range(24)), coeff=1.0)],
            target=1.0,
            target_type="charge",
            projector=ProjectorSpec(method="iao"),
        )
    ],
    solver=SolverOptions(mode="newton_kkt", conv_tol_constraint=1e-6),
)
e1 = mf.kernel()

print("Converged:", mf.converged)
print("Energy (Eh):", e0)
print("Constraint values:", mf.constraint_values())
print("Constraint residuals:", mf.constraint_residuals())
print("Lagrange Multipliers:", mf.v_lagrange)
