"""Population-analysis utilities for cDFT constraints."""

from __future__ import annotations

import numpy as np
from pyscf import lo


def lowdin_sqrt_overlap(overlap: np.ndarray) -> np.ndarray:
    """Compute a numerically stable square root of the overlap matrix."""
    eigvals, eigvecs = np.linalg.eigh(overlap)
    clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    return eigvecs @ np.diag(np.sqrt(clipped)) @ eigvecs.T


def atom_projector_from_aoslices(
    atom_indices: list[int], ao_slices: np.ndarray, nao: int
) -> np.ndarray:
    """Build AO-space projector matrix for a region of atoms."""
    projector = np.zeros((nao, nao))
    for atom_index in atom_indices:
        p0, p1 = int(ao_slices[atom_index, 2]), int(ao_slices[atom_index, 3])
        projector[p0:p1, p0:p1] = np.eye(p1 - p0)
    return projector


def lowdin_weight_matrix(
    overlap_sqrt: np.ndarray,
    atom_indices: list[int],
    ao_slices: np.ndarray,
) -> np.ndarray:
    """Construct Lowdin AO weight matrix for a region."""
    projector = atom_projector_from_aoslices(atom_indices, ao_slices, overlap_sqrt.shape[0])
    return overlap_sqrt @ projector @ overlap_sqrt


def orth_ao_weight_matrix(
    *,
    mol: object,
    basis: str,
    atom_indices: list[int],
    ao_slices: np.ndarray,
) -> np.ndarray:
    """Construct a region weight matrix in an orthogonal AO representation.

    This uses PySCF's ``pyscf.lo.orth_ao`` to build an orthogonal AO coefficient matrix
    ``C`` (AO -> orth-AO). The region operator is then

    ``W = C @ P @ C.T`` (or ``C @ P @ C.conj().T`` for complex),

    where ``P`` is a block-diagonal AO projector for the region atoms.
    """
    C = np.asarray(lo.orth_ao(mol, basis))
    projector = atom_projector_from_aoslices(atom_indices, ao_slices, C.shape[0])
    Ct = C.conj().T
    return C @ projector @ Ct


def constrained_population(total_density: np.ndarray, weight_matrix: np.ndarray) -> float:
    """Evaluate constrained-region population value."""
    value = np.trace(total_density @ weight_matrix)
    return float(np.real(value))
