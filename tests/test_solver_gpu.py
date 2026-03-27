"""GPU backend and solver-mode tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT
from cdft4pyscf.models import Constraint, FragmentTerm, RunRequest, SolverOptions


def test_gpu_backend_builds_or_raises_typed_error() -> None:
    """GPU requests should either build a CDFT wrapper or raise typed error."""
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


class DummyBaseMF:
    """Small fake mean-field object to unit-test solver mode wiring."""

    def __init__(self) -> None:
        self.mol = SimpleNamespace(
            atom_charges=lambda: [1.0, 1.0],
            aoslice_by_atom=lambda: np.array([[0, 0, 0, 1], [0, 0, 1, 2]], dtype=int),
        )
        self.converged = True

    def get_ovlp(self, _mol: Any) -> np.ndarray:
        """Return overlap matrix."""
        return np.eye(2, dtype=float)

    def get_hcore(self, _mol: Any) -> np.ndarray:
        """Return one-electron core matrix."""
        return np.eye(2, dtype=float)

    def get_veff(self, _mol: Any, _dm: Any = None) -> np.ndarray:
        """Return effective potential tensor."""
        return np.zeros((2, 2, 2), dtype=float)

    def eig(self, fock: np.ndarray, _overlap: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return a deterministic eigensystem."""
        return np.linalg.eigvalsh(fock[0]), np.eye(2, dtype=float)

    def get_occ(self, _mo_energy: np.ndarray, _mo_coeff: np.ndarray) -> np.ndarray:
        """Return fixed occupations."""
        return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=float)

    def make_rdm1(self, mo_coeff: Any = None, mo_occ: Any = None) -> np.ndarray:
        """Return a deterministic spin density."""
        _ = mo_coeff, mo_occ
        return np.array(
            [
                [[1.0, 0.0], [0.0, 0.0]],
                [[1.0, 0.0], [0.0, 0.0]],
            ],
            dtype=float,
        )

    def get_fock(self, *, h1e: Any = None, **_kwargs: Any) -> Any:
        """Return stacked alpha/beta Fock matrices."""
        return np.stack([h1e, h1e])

    def kernel(self, *_args: Any, **_kwargs: Any) -> float:
        """Return a fixed converged energy."""
        self.converged = True
        return -1.0


def test_newton_switches_solver_mode() -> None:
    """newton() should switch the wrapper into Newton-KKT mode."""
    mf = CDFT(
        DummyBaseMF(),
        constraints=[
            Constraint(
                name="n_a",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=1.0,
            )
        ],
        solver=SolverOptions(mode="micro"),
    )
    assert mf.solver_options.mode == "micro"
    mf.newton()
    assert mf.solver_options.mode == "newton_kkt"


def test_solver_fallback_records_used_mode() -> None:
    """Solver state should record which mode produced best residual."""
    mf = CDFT(
        DummyBaseMF(),
        constraints=[
            Constraint(
                name="n_a",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=1.0,
            )
        ],
        solver=SolverOptions(
            mode="outer_newton",
            fallback_modes=["outer_newton", "newton_kkt", "penalty", "micro"],
        ),
    )
    dm = mf.mf.make_rdm1()
    h1e = mf.mf.get_hcore(mf.mf.mol)
    s1e = mf.mf.get_ovlp(mf.mf.mol)
    vhf = mf.mf.get_veff(mf.mf.mol, dm)
    _ = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=0)
    assert mf.solver_state["fallback_used"] in {"outer_newton", "newton_kkt", "penalty", "micro"}
