"""Population-analysis utilities for cDFT constraints."""

from __future__ import annotations

import numpy as np

from cdft4pyscf.projectors import ao_selector
from cdft4pyscf.projectors import lowdin_sqrt_overlap as _lowdin_sqrt_overlap


def lowdin_sqrt_overlap(overlap: np.ndarray) -> np.ndarray:
    """Public compatibility wrapper for overlap square-root helper."""
    return _lowdin_sqrt_overlap(overlap)


def atom_projector_from_aoslices(
    atom_indices: list[int], ao_slices: np.ndarray, nao: int
) -> np.ndarray:
    """Build AO-space selector matrix for a region of atoms."""
    return ao_selector(atom_indices, ao_slices, nao)


def lowdin_weight_matrix(
    overlap_sqrt: np.ndarray, atom_indices: list[int], ao_slices: np.ndarray
) -> np.ndarray:
    """Construct Lowdin AO weight matrix for a region."""
    projector = atom_projector_from_aoslices(atom_indices, ao_slices, overlap_sqrt.shape[0])
    return overlap_sqrt @ projector @ overlap_sqrt


def constrained_population(total_density: np.ndarray, weight_matrix: np.ndarray) -> float:
    """Evaluate constrained-region population value."""
    value = np.trace(total_density @ weight_matrix)
    return float(np.real(value))
