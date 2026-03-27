#!/usr/bin/env python
"""Constrained DFT example using class-first kernel workflows."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, cast

import numpy as np
from pyscf import gto

from cdft4pyscf import CDFT_UKS_GPU, ConstraintSpec, RegionSpec
from cdft4pyscf.exceptions import BackendUnavailableError


def _load_gpu_uks() -> type[Any]:
    try:
        gpu_dft = importlib.import_module("gpu4pyscf.dft")
    except Exception as exc:  # pragma: no cover - runtime dependency guard.
        msg = "This example requires gpu4pyscf. Install the gpu/dev-gpu environment."
        raise BackendUnavailableError(msg) from exc
    return cast("type[Any]", gpu_dft.UKS)


UKS = _load_gpu_uks()

working_dir = Path(__file__).parent
chk_dir = working_dir / "chkfiles"
chk_dir.mkdir(exist_ok=True, parents=True)

Atom = tuple[str, float, float, float]

ANTHRACENE_ATOMS: list[Atom] = [
    ("C", -1.2251, -0.7061, 0.0001),
    ("C", -1.2250, 0.7060, 0.0001),
    ("C", 1.2251, -0.7061, 0.0001),
    ("C", 1.2251, 0.7061, 0.0002),
    ("C", 0.0000, -1.3938, 0.0000),
    ("C", 0.0000, 1.3937, 0.0001),
    ("C", -2.4505, -1.3930, 0.0000),
    ("C", -2.4504, 1.3930, -0.0001),
    ("C", 2.4505, -1.3929, 0.0000),
    ("C", 2.4505, 1.3929, 0.0000),
    ("C", -3.6588, -0.6955, -0.0001),
    ("C", -3.6587, 0.6956, -0.0001),
    ("C", 3.6587, -0.6956, -0.0002),
    ("C", 3.6587, 0.6956, -0.0002),
    ("H", 0.0000, -2.4839, -0.0001),
    ("H", 0.0000, 2.4838, 0.0000),
    ("H", -2.4744, -2.4809, 0.0000),
    ("H", -2.4742, 2.4808, -0.0001),
    ("H", 2.4743, -2.4808, 0.0000),
    ("H", 2.4742, 2.4808, 0.0000),
    ("H", -4.5991, -1.2391, -0.0002),
    ("H", -4.5989, 1.2394, -0.0003),
    ("H", 4.5989, -1.2393, -0.0004),
    ("H", 4.5989, 1.2393, -0.0003),
]

TCNE_ATOMS: list[Atom] = [
    ("N", 2.2122, -2.0265, 0.0000),
    ("N", -2.2208, -2.0177, 0.0000),
    ("N", -2.2117, 2.0274, 0.0001),
    ("N", 2.2209, 2.0168, 0.0001),
    ("C", -0.0016, -0.6769, 0.0000),
    ("C", 0.0014, 0.6769, -0.0002),
    ("C", 1.2225, -1.4199, -0.0001),
    ("C", -1.2290, -1.4147, 0.0002),
    ("C", -1.2226, 1.4200, -0.0001),
    ("C", 1.2288, 1.4145, 0.0000),
]

SPACING = [3.15]

BASIS_SET = "def2-tzvp"
XC = "b3lyp"
DISP = "d4"

for spacing in SPACING:
    tcne_atoms = [(*atom[:3], atom[3] + spacing) for atom in TCNE_ATOMS]
    atoms: list[Atom] = ANTHRACENE_ATOMS + tcne_atoms
    atom_lines = [f"{atom[0]} {atom[1]:.8f} {atom[2]:.8f} {atom[3]:.8f}" for atom in atoms]

    mol = gto.M(atom="\n".join(atom_lines), basis=BASIS_SET, charge=0, spin=0, verbose=4)

    mf = UKS(mol)
    mf.xc = XC
    mf.disp = DISP
    mf.chkfile = chk_dir / f"atcne_net_charge_{spacing}.chk"
    mf.init_guess = "chkfile"
    U0 = mf.kernel()

    mf_d0 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="anthracene_net_charge",
                region=RegionSpec(
                    name="anthracene",
                    atom_indices=list(range(len(ANTHRACENE_ATOMS))),
                ),
                kind="net_charge",
                target=0.0,
            ),
        ],
        initial_vc=[-0.107],
        population_basis="iao",
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d0.xc = XC
    mf_d0.max_cycle = 80
    mf_d0.disp = DISP
    mf_d0.chkfile = chk_dir / f"atcne_net_charge_{spacing}.chk"
    mf_d0.init_guess = "chkfile"
    mf_d0.verbose = 4
    E1 = mf_d0.kernel(mo_occ=mf.mo_occ, mo_coeff=mf.mo_coeff, dm=mf.make_rdm1())

    mf_d1 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="anthracene_net_charge",
                region=RegionSpec(
                    name="anthracene",
                    atom_indices=list(range(len(ANTHRACENE_ATOMS))),
                ),
                kind="net_charge",
                target=1.0,
            ),
        ],
        initial_vc=[0.01],
        population_basis="iao",
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d1.xc = XC
    mf_d1.max_cycle = 80
    mf_d1.disp = DISP
    mf_d1.chkfile = chk_dir / f"atcne_net_charge_{spacing}.chk"
    mf_d1.init_guess = "chkfile"
    mf_d1.verbose = 4
    E2 = mf_d1.kernel(mo_occ=mf.mo_occ, mo_coeff=mf.mo_coeff, dm=mf.make_rdm1())

    print(f"U0: {U0}")
    print(f"E1: {E1}")
    print(f"E2: {E2}")
    V = np.sqrt((U0 - E1) * (U0 - E2))
    print(f"V: {V}")
