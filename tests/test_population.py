"""Population operator tests."""

from __future__ import annotations

import numpy as np
import pytest

from cdft4pyscf.population import (
    constrained_population,
    lowdin_sqrt_overlap,
    lowdin_weight_matrix,
    orth_ao_weight_matrix,
)


def test_lowdin_sqrt_overlap_identity() -> None:
    """Identity overlap should map to identity square root."""
    overlap = np.eye(3)
    sqrt = lowdin_sqrt_overlap(overlap)
    np.testing.assert_allclose(sqrt, np.eye(3))


def test_lowdin_weight_and_population() -> None:
    """Region projector should isolate selected AO contribution."""
    overlap = np.eye(2)
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int)
    weight = lowdin_weight_matrix(lowdin_sqrt_overlap(overlap), [0], ao_slices)

    density = np.array([[1.2, 0.0], [0.0, 0.8]])
    expected = 1.2
    assert constrained_population(density, weight) == pytest.approx(expected)


@pytest.mark.parametrize("basis", ["lowdin", "meta_lowdin", "iao"])
def test_orth_ao_weight_matrix_matches_projected_orth_density(basis: str) -> None:
    """Orth-AO weight should match projector trace in orth basis."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto
    lo = pyscf.lo

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ao_slices = mol.aoslice_by_atom()
    atom_indices = [0]

    rng = np.random.default_rng(123)
    a = rng.normal(size=(mol.nao_nr(), mol.nao_nr()))
    density = (a + a.T) / 2.0

    weight = orth_ao_weight_matrix(
        mol=mol, basis=basis, atom_indices=atom_indices, ao_slices=ao_slices
    )

    C = np.asarray(lo.orth_ao(mol, basis))
    density_orth = C.conj().T @ density @ C
    p0, p1 = int(ao_slices[0, 2]), int(ao_slices[0, 3])
    projector = np.zeros_like(density_orth)
    projector[p0:p1, p0:p1] = np.eye(p1 - p0)
    expected = float(np.real(np.trace(projector @ density_orth)))

    got = constrained_population(density, weight)
    assert got == pytest.approx(expected, rel=1e-12, abs=1e-12)
