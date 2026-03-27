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

atom = (
    "C -1.2251 -0.7061 0.0001; C -1.2250 0.7060 0.0001; "
    "C 1.2251 -0.7061 0.0001; C 1.2251 0.7061 0.0002; "
    "N 2.2122 -2.0265 2.5000; N -2.2208 -2.0177 2.5000; "
    "N -2.2117 2.0274 2.5001; N 2.2209 2.0168 2.5001"
)

mol = gto.M(atom=atom, basis="def2-svp", charge=0, spin=0, verbose=4)
base = UKS(mol)
base.xc = "b3lyp"
base.max_cycle = 80
base.chkfile = chk_dir / "atcne_smoke.chk"

mf = CDFT(
    base,
    constraints=[
        Constraint(
            name="anthracene_charge",
            fragments=[FragmentTerm(atoms=[0, 1, 2, 3], coeff=1.0)],
            target=0.0,
            target_type="charge",
            projector=ProjectorSpec(method="lowdin"),
        )
    ],
    solver=SolverOptions(mode="micro", conv_tol_constraint=1e-6),
)

energy = mf.kernel()
print("Converged:", mf.converged)
print("Energy (Eh):", energy)
print("Constraint values:", mf.constraint_values())
print("Constraint residuals:", mf.constraint_residuals())
