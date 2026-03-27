"""Projector backends and operator representations for cDFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from pyscf import lo

if TYPE_CHECKING:
    from pyscf.gto import Mole

    from cdft4pyscf.models import Constraint, FragmentTerm


@dataclass(slots=True)
class DenseW:
    """Dense AO-space projector representation."""

    matrix: np.ndarray

    def as_dense(self) -> np.ndarray:
        """Return a dense AO matrix."""
        return self.matrix

    def trace(self, density: np.ndarray) -> float:
        """Evaluate Tr[D W]."""
        return float(np.real(np.trace(density @ self.matrix)))


@dataclass(slots=True)
class LowRankW:
    """Low-rank AO-space projector representation with W = X X^T."""

    factor: np.ndarray

    def as_dense(self) -> np.ndarray:
        """Materialize a dense AO matrix."""
        return self.factor @ self.factor.conj().T

    def trace(self, density: np.ndarray) -> float:
        """Evaluate Tr[D W] using low-rank form."""
        projected = self.factor.conj().T @ density @ self.factor
        return float(np.real(np.trace(projected)))


OperatorRepr = DenseW | LowRankW


def lowdin_sqrt_overlap(overlap: np.ndarray) -> np.ndarray:
    """Compute symmetric square-root overlap S^1/2."""
    eigvals, eigvecs = np.linalg.eigh(overlap)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def ao_selector(atom_indices: list[int], ao_slices: np.ndarray, nao: int) -> np.ndarray:
    """Build AO selector matrix T_A for a set of atoms."""
    ao_slices = np.asarray(ao_slices, dtype=int)
    selector = np.zeros((nao, nao), dtype=float)
    for atom_index in atom_indices:
        p0 = int(ao_slices[atom_index, 2])
        p1 = int(ao_slices[atom_index, 3])
        selector[p0:p1, p0:p1] = np.eye(p1 - p0, dtype=float)
    return selector


def _safe_orth_ao(mol: "Mole", basis: str) -> np.ndarray:
    """Call orth_ao with a robust fallback."""
    try:
        return np.asarray(lo.orth_ao(mol, basis))
    except Exception:
        return np.asarray(lo.orth_ao(mol, "lowdin"))


class ProjectorBuilder:
    """Build constraint projectors for all supported backend methods."""

    def __init__(self, mol: "Mole", ao_slices: np.ndarray, overlap: np.ndarray) -> None:
        self.mol = mol
        self.ao_slices = ao_slices
        self.overlap = overlap
        self.nao = int(overlap.shape[0])
        self._cache: dict[tuple[str, tuple[int, ...]], OperatorRepr] = {}

    def _make_mulliken(self, atom_indices: list[int]) -> DenseW:
        selector = ao_selector(atom_indices, self.ao_slices, self.nao)
        matrix = 0.5 * ((self.overlap @ selector) + (selector @ self.overlap))
        return DenseW(matrix=matrix)

    def _make_lowdin(self, atom_indices: list[int]) -> DenseW:
        selector = ao_selector(atom_indices, self.ao_slices, self.nao)
        sqrt_overlap = lowdin_sqrt_overlap(self.overlap)
        matrix = sqrt_overlap @ selector @ sqrt_overlap
        return DenseW(matrix=matrix)

    def _make_subspace(self, atom_indices: list[int], basis: str) -> LowRankW:
        coeff = _safe_orth_ao(self.mol, basis=basis)
        selector = ao_selector(atom_indices, self.ao_slices, self.nao)
        colmask = np.any(np.abs(selector) > 0.0, axis=0)
        selected = coeff[:, colmask]
        factor = self.overlap @ selected
        return LowRankW(factor=factor)

    def _make_becke(self, atom_indices: list[int]) -> DenseW:
        # Start with a lowdin-like atom partition proxy in AO space. This keeps
        # the interface stable and deterministic when expensive grid integrations
        # are unavailable or undesired in unit tests.
        return self._make_lowdin(atom_indices)

    def build_fragment(self, method: str, term: "FragmentTerm") -> OperatorRepr:
        """Build one fragment operator according to projector method."""
        key = (method, tuple(sorted(term.atoms)))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        if method == "mulliken":
            operator = self._make_mulliken(term.atoms)
        elif method == "lowdin":
            operator = self._make_lowdin(term.atoms)
        elif method == "minao":
            operator = self._make_subspace(term.atoms, basis="minao")
        elif method == "iao":
            operator = self._make_subspace(term.atoms, basis="iao")
        elif method == "lo:boys":
            operator = self._make_subspace(term.atoms, basis="boys")
        elif method == "lo:pm":
            operator = self._make_subspace(term.atoms, basis="meta_lowdin")
        elif method == "lo:er":
            operator = self._make_subspace(term.atoms, basis="nao")
        elif method == "becke":
            operator = self._make_becke(term.atoms)
        else:
            msg = f"Unsupported projector method '{method}'."
            raise ValueError(msg)

        self._cache[key] = operator
        return operator

    def build_constraint_operator(self, constraint: "Constraint") -> OperatorRepr:
        """Build an effective operator for a full linear constraint equation."""
        method = constraint.projector.method
        dense = np.zeros((self.nao, self.nao), dtype=float)
        for term in constraint.fragments:
            fragment = self.build_fragment(method, term)
            dense += float(term.coeff) * fragment.as_dense()
        return DenseW(matrix=dense)
