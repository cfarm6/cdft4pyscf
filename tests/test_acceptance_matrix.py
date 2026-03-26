"""Acceptance matrix tests for representative combinations."""

from __future__ import annotations

import pytest

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT_UKS, CDFT_UKS_GPU
from cdft4pyscf.models import ConstraintSpec, RegionSpec, RunRequest


@pytest.mark.parametrize(
    "xc",
    [
        "lda,vwn",
        "pbe,pbe",
    ],
)
def test_acceptance_matrix_cpu_cases(xc: str) -> None:
    """Small matrix over functionals on CPU backend."""
    pytest.importorskip("pyscf")
    request = RunRequest(
        atom="H 0 0 0; H 0 0 0.74",
        backend="cpu",
        xc=xc,
        constraints=[
            ConstraintSpec(
                name="q_tot",
                kind="net_charge",
                target=0.0,
                region=RegionSpec(name="all_atoms", atom_indices=[0, 1]),
            )
        ],
    )
    mf = build_cdft_mean_field(request)
    assert isinstance(mf, CDFT_UKS)
    assert mf.xc == xc


def test_acceptance_matrix_gpu_case() -> None:
    """GPU matrix case verifies typed backend error or successful construction."""
    request = RunRequest(
        atom="H 0 0 0; H 0 0 0.74",
        backend="gpu",
        constraints=[
            ConstraintSpec(
                name="q_tot",
                kind="net_charge",
                target=0.0,
                region=RegionSpec(name="all_atoms", atom_indices=[0, 1]),
            )
        ],
    )
    try:
        mf = build_cdft_mean_field(request)
    except BackendUnavailableError:
        return
    assert isinstance(mf, CDFT_UKS_GPU)
