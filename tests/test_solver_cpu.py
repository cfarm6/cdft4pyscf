"""CPU constrained mean-field tests."""

from __future__ import annotations

import pytest

from cdft4pyscf.meanfield import CDFT_UKS
from cdft4pyscf.models import ConstraintSpec, RegionSpec


def test_cdft_uks_builds_constraint_accessors() -> None:
    """Class API should expose values/residuals/multipliers by constraint name."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    mf = CDFT_UKS(
        mol,
        constraints=[
            ConstraintSpec(
                name="n_frag_a",
                kind="electron_number",
                target=1.0,
                region=RegionSpec(name="frag_a", atom_indices=[0]),
            )
        ],
        population_basis="lowdin",
    )
    multipliers = mf.multiplier_by_constraint()
    values = mf.constraint_values(dm=mf.get_init_guess(mol))
    residuals = mf.constraint_residuals(dm=mf.get_init_guess(mol))

    assert "n_frag_a" in multipliers
    assert "n_frag_a" in values
    assert "n_frag_a" in residuals


def test_cdft_get_fock_updates_vc() -> None:
    """CDFT get_fock should run multiplier micro-optimization during SCF cycles."""
    pyscf = pytest.importorskip("pyscf")
    dft = pyscf.dft
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    ref_mf = dft.UKS(mol)
    dm = ref_mf.get_init_guess(mol)
    h1e = ref_mf.get_hcore(mol)
    s1e = ref_mf.get_ovlp(mol)
    vhf = ref_mf.get_veff(mol, dm)

    mf = CDFT_UKS(
        mol,
        constraints=[
            ConstraintSpec(
                name="q_tot",
                kind="net_charge",
                target=0.0,
                region=RegionSpec(name="all_atoms", atom_indices=[0, 1]),
            )
        ],
        initial_vc=[0.123],
        population_basis="lowdin",
    )
    mf.xc = "lda,vwn"
    assert mf.vc[0] == pytest.approx(0.123)

    fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=0)
    assert fock.shape == (2, h1e.shape[0], h1e.shape[1])
    assert mf.cdft_inner_calls > 0
