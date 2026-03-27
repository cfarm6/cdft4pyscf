#!/usr/bin/env python
"""Constrained DFT example using class-first kernel workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, cast

import numpy as np
from gpu4pyscf.dft import UKS
from pyscf import gto

from cdft4pyscf import CDFT_UKS_GPU, ConstraintSpec, RegionSpec


class _HasMultiplierByConstraint(Protocol):
    def multiplier_by_constraint(self) -> dict[str, float]: ...


working_dir = Path(__file__).parent
chk_dir = working_dir / "chkfiles"
chk_dir.mkdir(exist_ok=True, parents=True)
structure_dir = working_dir / "structures"

BASIS_SET = "def2-svp"
XC = "b3lyp"
DISP = "d3bj"
V0_init = [0.001]
V1_init = [0.001]

for structure_file in list(structure_dir.glob("*.xyz"))[:1]:
    spacing = structure_file.stem.split("=")[1]
    mol = gto.M(atom=structure_file.as_posix(), basis=BASIS_SET, charge=0, spin=0, verbose=4)

    mf = UKS(mol)
    mf.xc = XC
    mf.disp = DISP
    mf.chkfile = chk_dir / f"U0_spacing={spacing}.chk"
    mf.init_guess = "chkfile"
    U0 = mf.kernel()

    mf_d0 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="anthracene_net_charge",
                region=RegionSpec(
                    name="anthracene",
                    atom_indices=list(range(24)),
                ),
                kind="net_charge",
                target=0.0,
            ),
        ],
        vc_max_step=0.001,
        population_basis="meta_lowdin",
        initial_vc=V0_init,
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d0.xc = XC
    mf_d0.max_cycle = 80
    mf_d0.disp = DISP
    mf_d0.chkfile = chk_dir / f"E0_spacing={spacing}.chk"
    # mf_d0.init_guess = "chkfile"
    E0 = mf_d0.kernel()
    V0_init = [
        cast("_HasMultiplierByConstraint", mf_d0).multiplier_by_constraint()[
            "anthracene_net_charge"
        ]
    ]

    mf_d1 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="anthracene_net_charge",
                region=RegionSpec(
                    name="anthracene",
                    atom_indices=list(range(24)),
                ),
                kind="net_charge",
                target=1.0,
            ),
        ],
        population_basis="meta_lowdin",
        initial_vc=V1_init,
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d1.xc = XC
    mf_d1.max_cycle = 80
    mf_d1.disp = DISP
    mf_d1.chkfile = chk_dir / f"E1_spacing={spacing}.chk"
    # mf_d1.init_guess = "chkfile"
    E1 = mf_d1.kernel()
    V1_init = [
        cast("_HasMultiplierByConstraint", mf_d1).multiplier_by_constraint()[
            "anthracene_net_charge"
        ]
    ]
    print(f"U0: {U0}")
    print(f"E0: {E0}")
    print(f"E1: {E1}")
    print(f"V0: {V0_init}")
    print(f"V1: {V1_init}")
    print(f"V: {np.sqrt((U0 - E0) * (U0 - E1))}")
