"""CPU constrained mean-field tests."""

from __future__ import annotations

import pytest

from cdft4pyscf.meanfield import CDFT
from cdft4pyscf.models import Constraint, FragmentTerm, ProjectorSpec, SolverOptions


def test_cdft_wrapper_builds_constraint_accessors() -> None:
    """Wrapper API should expose values and residuals by constraint name."""
    pyscf = pytest.importorskip("pyscf")
    dft, gto = pyscf.dft, pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    base = dft.UKS(mol)
    mf = CDFT(
        base,
        constraints=[
            Constraint(
                name="n_frag_a",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=1.0,
                target_type="electrons",
                projector=ProjectorSpec(method="lowdin"),
            )
        ],
        solver=SolverOptions(mode="micro"),
    )
    multipliers = mf.v_lagrange
    values = mf.constraint_values(dm=mf.get_init_guess(mol))
    residuals = mf.constraint_residuals(dm=mf.get_init_guess(mol))

    assert multipliers.shape == (1,)
    assert "n_frag_a" in values
    assert "n_frag_a" in residuals


def test_cdft_get_fock_updates_v_lagrange() -> None:
    """CDFT get_fock should run multiplier updates during SCF cycles."""
    pyscf = pytest.importorskip("pyscf")
    dft = pyscf.dft
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    ref_mf = dft.UKS(mol)
    dm = ref_mf.get_init_guess(mol)
    h1e = ref_mf.get_hcore(mol)
    s1e = ref_mf.get_ovlp(mol)
    vhf = ref_mf.get_veff(mol, dm)

    base = dft.UKS(mol)
    mf = CDFT(
        base,
        constraints=[
            Constraint(
                name="q_tot",
                fragments=[FragmentTerm(atoms=[0, 1], coeff=1.0)],
                target=0.0,
                target_type="charge",
                projector=ProjectorSpec(method="lowdin"),
            )
        ],
        solver=SolverOptions(initial_v_lagrange=[0.123], mode="micro"),
    )
    mf.xc = "lda,vwn"
    assert mf.v_lagrange[0] == pytest.approx(0.123)

    fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=0)
    assert fock.shape == (2, h1e.shape[0], h1e.shape[1])
    assert mf.solver_state["inner_solver_evaluations"] > 0
