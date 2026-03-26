"""Compatibility numerical helpers for constrained mean-field classes."""

from __future__ import annotations

from typing import Any

import numpy as np


def diagonalize_fock(
    mf: Any, fock: np.ndarray, overlap: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Diagonalize alpha/beta Fock blocks and return MO energies and coeffs."""
    mo_energy, mo_coeff = mf.eig(fock, overlap)
    return np.asarray(mo_energy), np.asarray(mo_coeff)


def density_from_fock(mf: Any, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
    """Build UKS density matrix from alpha/beta Fock blocks."""
    mo_energy, mo_coeff = diagonalize_fock(mf, fock, overlap)
    mo_occ = mf.get_occ(mo_energy, mo_coeff)
    return mf.make_rdm1(mo_coeff, mo_occ)
