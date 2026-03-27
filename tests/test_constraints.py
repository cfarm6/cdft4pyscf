"""Constraint assembly tests."""

from __future__ import annotations

import numpy as np
import pytest

from cdft4pyscf.constraints import (
    build_constraint_system,
    evaluate_constraint_residuals,
    evaluate_constraint_values,
    report_constraint_residuals,
    report_constraint_values,
)
from cdft4pyscf.models import Constraint, FragmentTerm, ProjectorSpec


def test_build_constraint_system_mixed_constraints() -> None:
    """Electron and net-charge constraints can coexist."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        Constraint(
            name="n_a",
            fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
            target=1.0,
            target_type="electrons",
            projector=ProjectorSpec(method="lowdin"),
        ),
        Constraint(
            name="q_a",
            fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
            target=0.0,
            target_type="charge",
            projector=ProjectorSpec(method="lowdin"),
        ),
    ]

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        overlap=overlap,
    )

    assert system.names == ["n_a", "q_a"]
    np.testing.assert_allclose(system.targets, np.array([1.0, 1.0]))


def test_build_constraint_system_supports_difference_constraint() -> None:
    """Signed fragment coefficients build one effective difference operator."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        Constraint(
            name="delta_ab",
            fragments=[
                FragmentTerm(atoms=[0], coeff=1.0),
                FragmentTerm(atoms=[1], coeff=-1.0),
            ],
            target=0.0,
            projector=ProjectorSpec(method="mulliken"),
        )
    ]

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        overlap=overlap,
    )
    assert system.targets.shape == (1,)
    assert system.operators[0].as_dense().shape == (mol.nao_nr(), mol.nao_nr())


def test_build_constraint_system_rejects_out_of_bounds_atom_index() -> None:
    """Constraint atom indices must be within molecule atom range."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        Constraint(
            name="q_bad",
            fragments=[FragmentTerm(atoms=[2], coeff=1.0)],
            target=0.0,
            target_type="charge",
        )
    ]

    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    with pytest.raises(ValueError, match="outside molecule atom range"):
        build_constraint_system(
            constraints=constraints,
            mol=mol,
            ao_slices=mol.aoslice_by_atom(),
            atom_charges=np.asarray(mol.atom_charges(), dtype=float),
            overlap=overlap,
        )


def test_net_charge_maps_to_regional_lowdin_partial_charge() -> None:
    """Net-charge uses a regional electron target equivalent to Z_region - q_target."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        Constraint(
            name="q_a",
            fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
            target=0.4,
            target_type="charge",
            projector=ProjectorSpec(method="lowdin"),
        )
    ]
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=atom_charges,
        overlap=overlap,
    )
    np.testing.assert_allclose(system.targets, np.array([0.6]))

    # Numerical region_charge validation lives in population-operator tests.


def test_reporting_converts_net_charge_to_partial_charge() -> None:
    """Reporting maps internal electron counts to net-charge values."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        Constraint(
            name="n_a",
            fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
            target=1.0,
            target_type="electrons",
            projector=ProjectorSpec(method="lowdin"),
        ),
        Constraint(
            name="q_a",
            fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
            target=0.4,
            target_type="charge",
            projector=ProjectorSpec(method="lowdin"),
        ),
    ]
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)
    density = np.eye(mol.nao_nr(), dtype=float)
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=atom_charges,
        overlap=overlap,
    )
    raw_values = evaluate_constraint_values(density, system)
    shown_values = report_constraint_values(raw_values, system)
    assert shown_values.shape == (2,)

    raw_residuals = evaluate_constraint_residuals(density, system)
    shown_residuals = report_constraint_residuals(raw_residuals, system)
    assert shown_residuals.shape == (2,)
