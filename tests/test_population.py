"""Population operator tests."""

from __future__ import annotations

import numpy as np
import pytest

from cdft4pyscf.models import FragmentTerm
from cdft4pyscf.population import constrained_population, lowdin_sqrt_overlap, lowdin_weight_matrix
from cdft4pyscf.projectors import DenseW, LowRankW, ProjectorBuilder, ao_selector


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
def test_projector_builder_builds_supported_methods(basis: str) -> None:
    """Builder should create valid projector operators for configured methods."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ao_slices = mol.aoslice_by_atom()
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    builder = ProjectorBuilder(mol=mol, ao_slices=ao_slices, overlap=overlap)

    method = "lowdin" if basis in {"lowdin", "meta_lowdin"} else "iao"
    operator = builder.build_fragment(method, FragmentTerm(atoms=[0], coeff=1.0))
    assert operator.as_dense().shape == (mol.nao_nr(), mol.nao_nr())
    assert np.allclose(operator.as_dense(), operator.as_dense().T, atol=1e-10)


def test_dense_and_lowrank_trace_consistency() -> None:
    """Dense and low-rank forms should give consistent traces."""
    rng = np.random.default_rng(10)
    density = rng.normal(size=(4, 4))
    density = (density + density.T) / 2.0
    factor = rng.normal(size=(4, 2))

    low_rank = LowRankW(factor=factor)
    dense = DenseW(matrix=low_rank.as_dense())
    assert low_rank.trace(density) == pytest.approx(dense.trace(density))


def test_ao_selector_projects_atom_block() -> None:
    """AO selector should mark exactly the selected atomic block."""
    ao_slices = np.array([[0, 0, 0, 1], [0, 0, 1, 3]], dtype=int)
    selector = ao_selector([1], ao_slices, nao=3)
    expected = np.diag([0.0, 1.0, 1.0])
    np.testing.assert_allclose(selector, expected)


def test_becke_projector_shape_and_trace_sanity() -> None:
    """Becke backend should produce a valid AO operator and finite trace."""
    pyscf = pytest.importorskip("pyscf")
    gto = pyscf.gto

    mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    ao_slices = mol.aoslice_by_atom()
    overlap = np.asarray(mol.intor_symmetric("int1e_ovlp"), dtype=float)
    builder = ProjectorBuilder(mol=mol, ao_slices=ao_slices, overlap=overlap)

    operator = builder.build_fragment("becke", FragmentTerm(atoms=[0], coeff=1.0))
    matrix = operator.as_dense()
    assert matrix.shape == (mol.nao_nr(), mol.nao_nr())
    assert np.allclose(matrix, matrix.T, atol=1e-10)
    density = np.eye(mol.nao_nr(), dtype=float)
    assert np.isfinite(constrained_population(density, matrix))
