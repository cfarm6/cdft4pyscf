"""Constraint assembly and residual evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from cdft4pyscf.population import constrained_population, lowdin_sqrt_overlap, lowdin_weight_matrix

if TYPE_CHECKING:
    from cdft4pyscf.models import ConstraintSpec


@dataclass(slots=True)
class ConstraintSystem:
    """Linearized representation of constraints used by the solver."""

    names: list[str]
    kinds: list[str]
    targets: np.ndarray
    operators: list[np.ndarray]
    report_scales: np.ndarray
    report_offsets: np.ndarray


def build_constraint_system(
    *,
    constraints: list["ConstraintSpec"],
    overlap: np.ndarray,
    ao_slices: np.ndarray,
    atom_charges: np.ndarray,
) -> ConstraintSystem:
    """Build per-constraint operators and target vector."""
    overlap_sqrt = lowdin_sqrt_overlap(overlap)

    names: list[str] = []
    kinds: list[str] = []
    targets: list[float] = []
    operators: list[np.ndarray] = []
    report_scales: list[float] = []
    report_offsets: list[float] = []

    for constraint in constraints:
        names.append(constraint.name)
        kinds.append(constraint.kind)
        if constraint.kind == "electron_number":
            region_spec = constraint.region
            if isinstance(region_spec, list):
                atom_indices = list(
                    dict.fromkeys(
                        atom_index for region in region_spec for atom_index in region.atom_indices
                    )
                )
            else:
                atom_indices = region_spec.atom_indices if region_spec is not None else []
            operator = lowdin_weight_matrix(overlap_sqrt, atom_indices, ao_slices)
            targets.append(constraint.target)
            operators.append(operator)
            report_scales.append(1.0)
            report_offsets.append(0.0)
        elif constraint.kind == "net_charge":
            region_spec = constraint.region
            if region_spec is None or isinstance(region_spec, list):
                msg = "net_charge constraints require a single region."
                raise ValueError(msg)
            atom_indices = region_spec.atom_indices
            operator = lowdin_weight_matrix(overlap_sqrt, atom_indices, ao_slices)
            region_nuclear_charge = float(np.sum(atom_charges[atom_indices], dtype=float))
            target_electrons = region_nuclear_charge - constraint.target
            targets.append(target_electrons)
            operators.append(operator)
            report_scales.append(-1.0)
            report_offsets.append(region_nuclear_charge)
        else:
            msg = f"Unsupported constraint kind '{constraint.kind}'."
            raise ValueError(msg)

    return ConstraintSystem(
        names=names,
        kinds=kinds,
        targets=np.asarray(targets, dtype=float),
        operators=operators,
        report_scales=np.asarray(report_scales, dtype=float),
        report_offsets=np.asarray(report_offsets, dtype=float),
    )


def evaluate_constraint_values(total_density: np.ndarray, system: ConstraintSystem) -> np.ndarray:
    """Evaluate all constraint values for the current total density."""
    return np.asarray(
        [constrained_population(total_density, operator) for operator in system.operators],
        dtype=float,
    )


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
