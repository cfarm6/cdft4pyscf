"""Constrained DFT example using class-first kernel workflows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from gpu4pyscf.dft import UKS
from pyscf import gto
from pyscf.lib import chkfile

from cdft4pyscf import CDFT_UKS_GPU, ConstraintSpec, RegionSpec

working_dir = Path(__file__).parent
chk_dir = working_dir / "chkfiles"
chk_dir.mkdir(exist_ok=True, parents=True)

DONOR_ATOMS = [0, 1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 37]
BASIS_SET = "def2-svp"
XC = "b3lyp"
DISP = "d3bj"
INPUT_STRUCTURE = working_dir / "fa-aq.xyz"


def main() -> None:
    """Run a small constrained DFT calculation."""
    mol = gto.M(atom=INPUT_STRUCTURE.as_posix(), basis=BASIS_SET, charge=0, spin=0, verbose=4)
    mf = UKS(mol)
    mf.xc = XC
    mf.disp = DISP
    mf.chkfile = chk_dir / "fa-aq.chk"
    mf.init_guess = "chkfile"
    U0 = mf.kernel()
    ## Neutral Constraint
    mf_d0 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="donor_region",
                kind="net_charge",
                target=0.0,
                region=RegionSpec(
                    name="donor_region",
                    atom_indices=DONOR_ATOMS,
                ),
            )
        ],
        population_basis="meta_lowdin",
        initial_vc=[0.00030474565563085146],
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d0.xc = XC
    mf_d0.max_cycle = 80
    mf_d0.disp = DISP
    mf_d0.chkfile = chk_dir / "D0_fa-aq.chk"
    mf_d0.init_guess = "chkfile"
    mf_d0.verbose = 4
    E1 = mf_d0.kernel()

    ## Positive Constraint
    mf_d0 = CDFT_UKS_GPU(
        mol,
        constraints=[
            ConstraintSpec(
                name="donor_region",
                kind="net_charge",
                target=1.0,
                region=RegionSpec(
                    name="donor_region",
                    atom_indices=DONOR_ATOMS,
                ),
            )
        ],
        population_basis="meta_lowdin",
        initial_vc=[-0.005],
        conv_tol=1e-7,
        vc_tol=1e-7,
        vc_max_cycle=80,
        log_inner_solver=True,
    )
    mf_d0.xc = XC
    mf_d0.max_cycle = 80
    mf_d0.disp = DISP
    mf_d0.chkfile = chk_dir / "D1_fa-aq.chk"
    # mf_d0.init_guess = "chkfile"
    mf_d0.__dict__.update(chkfile.load(chk_dir / "fa-aq.chk", "scf"))
    mf_d0.verbose = 4
    E2 = mf_d0.kernel()
    ###
    print(f"U0: {U0}")
    print(f"E1: {E1}")
    print(f"E2: {E2}")
    V = np.sqrt((U0 - E1) * (U0 - E2))
    print(f"V: {V}")


if __name__ == "__main__":
    main()
