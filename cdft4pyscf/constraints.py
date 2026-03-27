"""Constraint assembly and residual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cdft4pyscf.projectors import OperatorRepr, ProjectorBuilder

if TYPE_CHECKING:
    from pyscf.gto import Mole

    from cdft4pyscf.models import Constraint


@dataclass(slots=True)
class ConstraintSystem:
    """Linearized representation of constraints used by the solver."""

    names: list[str]
    targets: np.ndarray
    target_types: list[str]
    operators: list[OperatorRepr]
    report_scales: np.ndarray
    report_offsets: np.ndarray


def build_constraint_system(
    *,
    constraints: list["Constraint"],
    mol: "Mole",
    ao_slices: np.ndarray,
    atom_charges: np.ndarray,
    overlap: np.ndarray,
) -> ConstraintSystem:
    """Build per-constraint operators and target vector."""
    n_atoms = int(np.asarray(atom_charges, dtype=float).shape[0])
    builder = ProjectorBuilder(mol=mol, ao_slices=ao_slices, overlap=overlap)

    def _validate_atom_indices(indices: list[int], *, constraint_name: str) -> None:
        invalid = sorted({int(index) for index in indices if int(index) >= n_atoms})
        if invalid:
            msg = (
                f"Constraint '{constraint_name}' references atom indices {invalid} "
                f"outside molecule atom range [0, {n_atoms - 1}]."
            )
            raise ValueError(msg)

    names: list[str] = []
    targets: list[float] = []
    target_types: list[str] = []
    operators: list[OperatorRepr] = []
    report_scales: list[float] = []
    report_offsets: list[float] = []

    for constraint in constraints:
        names.append(constraint.name)
        target_types.append(constraint.target_type)

        atom_indices_flat = [atom for term in constraint.fragments for atom in term.atoms]
        _validate_atom_indices(atom_indices_flat, constraint_name=constraint.name)
        operator = builder.build_constraint_operator(constraint)
        operators.append(operator)

        if constraint.target_type == "electrons":
            targets.append(float(constraint.target))
            report_scales.append(1.0)
            report_offsets.append(0.0)
        elif constraint.target_type == "charge":
            weighted_nuclear_charge = 0.0
            for term in constraint.fragments:
                z_fragment = float(np.sum(atom_charges[term.atoms], dtype=float))
                weighted_nuclear_charge += float(term.coeff) * z_fragment
            target_electrons = weighted_nuclear_charge - float(constraint.target)
            targets.append(float(target_electrons))
            report_scales.append(-1.0)
            report_offsets.append(weighted_nuclear_charge)
        else:
            msg = f"Unsupported target_type '{constraint.target_type}'."
            raise ValueError(msg)

    return ConstraintSystem(
        names=names,
        targets=np.asarray(targets, dtype=float),
        target_types=target_types,
        operators=operators,
        report_scales=np.asarray(report_scales, dtype=float),
        report_offsets=np.asarray(report_offsets, dtype=float),
    )


def evaluate_constraint_values(total_density: np.ndarray, system: ConstraintSystem) -> np.ndarray:
    """Evaluate all constraint values for the current total density."""
    return np.asarray([operator.trace(total_density) for operator in system.operators], dtype=float)


def evaluate_constraint_residuals(
    total_density: np.ndarray, system: ConstraintSystem
) -> np.ndarray:
    """Evaluate residual vector value-target."""
    values = evaluate_constraint_values(total_density, system)
    return values - system.targets


def report_constraint_values(raw_values: np.ndarray, system: ConstraintSystem) -> np.ndarray:
    """Convert internal solver values to user-facing values per constraint kind."""
    values = np.asarray(raw_values, dtype=float)
    return system.report_offsets + (system.report_scales * values)


def report_constraint_residuals(raw_residuals: np.ndarray, system: ConstraintSystem) -> np.ndarray:
    """Convert internal solver residuals to user-facing residuals per kind."""
    residuals = np.asarray(raw_residuals, dtype=float)
    return system.report_scales * residuals
