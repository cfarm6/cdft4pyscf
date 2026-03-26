"""Population operator tests."""

from __future__ import annotations

import numpy as np
import pytest

from cdft4pyscf.population import constrained_population, lowdin_sqrt_overlap, lowdin_weight_matrix


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
