"""Acceptance matrix tests for representative combinations."""

from __future__ import annotations

import pytest

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT
from cdft4pyscf.models import Constraint, FragmentTerm, RunRequest


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
            Constraint(
                name="q_tot",
                fragments=[FragmentTerm(atoms=[0, 1], coeff=1.0)],
                target=0.0,
                target_type="charge",
            )
        ],
    )
    mf = build_cdft_mean_field(request)
    assert isinstance(mf, CDFT)
    assert mf.xc == xc


def test_acceptance_matrix_gpu_case() -> None:
    """GPU matrix case verifies typed backend error or successful construction."""
    request = RunRequest(
        atom="H 0 0 0; H 0 0 0.74",
        backend="gpu",
        constraints=[
            Constraint(
                name="q_tot",
                fragments=[FragmentTerm(atoms=[0, 1], coeff=1.0)],
                target=0.0,
                target_type="charge",
            )
        ],
    )
    try:
        mf = build_cdft_mean_field(request)
    except BackendUnavailableError:
        return
    assert isinstance(mf, CDFT)
