"""Constraint assembly tests."""

from __future__ import annotations

import numpy as np

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
    overlap = np.eye(2)
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int)

    system = build_constraint_system(
        constraints=constraints,
        overlap=overlap,
        ao_slices=ao_slices,
        atom_charges=np.array([1.0, 1.0]),
    )

    assert system.names == ["n_a", "q_a"]
    np.testing.assert_allclose(system.targets, np.array([1.0, 1.0]))

    density = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = evaluate_constraint_values(density, system)
    np.testing.assert_allclose(values, np.array([1.0, 1.0]))


def test_build_constraint_system_combines_region_list() -> None:
    """Electron-number constraint list regions combine as one operator."""
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
    overlap = np.eye(2)
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int)

    system = build_constraint_system(
        constraints=constraints,
        overlap=overlap,
        ao_slices=ao_slices,
        atom_charges=np.array([1.0, 1.0]),
    )
    density = np.array([[1.0, 0.0], [0.0, 1.0]])
    values = evaluate_constraint_values(density, system)
    np.testing.assert_allclose(values, np.array([2.0]))


def test_net_charge_maps_to_regional_lowdin_partial_charge() -> None:
    """Net-charge uses a regional electron target equivalent to Z_region - q_target."""
    constraints = [
        ConstraintSpec(
            name="q_a",
            kind="net_charge",
            target=0.4,
            region=RegionSpec(name="a", atom_indices=[0]),
        )
    ]
    overlap = np.eye(2)
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int)
    atom_charges = np.array([1.0, 1.0])

    system = build_constraint_system(
        constraints=constraints,
        overlap=overlap,
        ao_slices=ao_slices,
        atom_charges=atom_charges,
    )
    np.testing.assert_allclose(system.targets, np.array([0.6]))

    density = np.array([[0.6, 0.0], [0.0, 1.0]])
    values = evaluate_constraint_values(density, system)
    region_charge = atom_charges[0] - values[0]
    np.testing.assert_allclose(region_charge, np.array(0.4))


def test_reporting_converts_net_charge_to_partial_charge() -> None:
    """Reporting maps internal electron counts to net-charge values."""
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
    overlap = np.eye(2)
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int)
    atom_charges = np.array([1.0, 1.0])
    density = np.array([[0.6, 0.0], [0.0, 1.0]])

    system = build_constraint_system(
        constraints=constraints,
        overlap=overlap,
        ao_slices=ao_slices,
        atom_charges=atom_charges,
    )
    raw_values = evaluate_constraint_values(density, system)
    shown_values = report_constraint_values(raw_values, system)
    np.testing.assert_allclose(shown_values, np.array([0.6, 0.4]))

    raw_residuals = evaluate_constraint_residuals(density, system)
    shown_residuals = report_constraint_residuals(raw_residuals, system)
    np.testing.assert_allclose(shown_residuals, np.array([-0.4, 0.0]))
