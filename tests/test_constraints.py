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
from cdft4pyscf.models import ConstraintSpec, RegionSpec


def test_build_constraint_system_mixed_constraints() -> None:
    """Electron-number and net-charge constraints can coexist."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        ConstraintSpec(
            name="n_a",
            kind="electron_number",
            target=1.0,
            region=RegionSpec(name="a", atom_indices=[0]),
        ),
        ConstraintSpec(
            name="q_a",
            kind="net_charge",
            target=0.0,
            region=RegionSpec(name="a", atom_indices=[0]),
        ),
    ]

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
    )

    assert system.names == ["n_a", "q_a"]
    np.testing.assert_allclose(system.targets, np.array([1.0, 1.0]))


def test_build_constraint_system_combines_region_list() -> None:
    """Electron-number constraint list regions combine as one operator."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        ConstraintSpec(
            name="n_ab",
            kind="electron_number",
            target=2.0,
            region=[
                RegionSpec(name="a", atom_indices=[0]),
                RegionSpec(name="b", atom_indices=[1]),
            ],
        )
    ]

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=np.asarray(mol.atom_charges(), dtype=float),
    )
    assert system.targets.shape == (1,)


def test_net_charge_maps_to_regional_lowdin_partial_charge() -> None:
    """Net-charge uses a regional electron target equivalent to Z_region - q_target."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        ConstraintSpec(
            name="q_a",
            kind="net_charge",
            target=0.4,
            region=RegionSpec(name="a", atom_indices=[0]),
        )
    ]
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=atom_charges,
    )
    np.testing.assert_allclose(system.targets, np.array([0.6]))

    # Numerical region_charge validation lives in population-operator tests.


def test_reporting_converts_net_charge_to_partial_charge() -> None:
    """Reporting maps internal electron counts to net-charge values."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", spin=0, charge=0, verbose=0)
    constraints = [
        ConstraintSpec(
            name="n_a",
            kind="electron_number",
            target=1.0,
            region=RegionSpec(name="a", atom_indices=[0]),
        ),
        ConstraintSpec(
            name="q_a",
            kind="net_charge",
            target=0.4,
            region=RegionSpec(name="a", atom_indices=[0]),
        ),
    ]
    atom_charges = np.asarray(mol.atom_charges(), dtype=float)
    density = np.eye(mol.nao_nr(), dtype=float)

    system = build_constraint_system(
        constraints=constraints,
        mol=mol,
        ao_slices=mol.aoslice_by_atom(),
        atom_charges=atom_charges,
    )
    raw_values = evaluate_constraint_values(density, system)
    shown_values = report_constraint_values(raw_values, system)
    assert shown_values.shape == (2,)

    raw_residuals = evaluate_constraint_residuals(density, system)
    shown_residuals = report_constraint_residuals(raw_residuals, system)
    assert shown_residuals.shape == (2,)
