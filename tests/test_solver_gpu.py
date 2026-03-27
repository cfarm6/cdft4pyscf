"""GPU backend tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np

from cdft4pyscf import meanfield
from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.constraints import ConstraintSystem
from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT_UKS_GPU, _CDFTMixin
from cdft4pyscf.models import ConstraintSpec, RegionSpec, RunRequest

UNCHANGED_VC = 0.1
BOUNDED_VC = 0.75


def test_gpu_backend_builds_or_raises_typed_error() -> None:
    """GPU requests should either build GPU CDFT or raise typed backend error."""
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


def test_get_fock_avoids_implicit_numpy_coercion() -> None:
    """get_fock should not coerce GPU-like arrays through np.asarray."""

    class FakeGPUArray:
        def __init__(self, data: np.ndarray) -> None:
            self._data = np.asarray(data, dtype=float)

        def __getitem__(self, index):
            return FakeGPUArray(self._data[index])

        def __add__(self, other):
            if isinstance(other, FakeGPUArray):
                return FakeGPUArray(self._data + other._data)
            return FakeGPUArray(self._data + np.asarray(other, dtype=float))

        def __radd__(self, other):
            return self.__add__(other)

        def get(self) -> np.ndarray:
            return self._data.copy()

        def __array__(self, dtype=None):  # pragma: no cover - failure path sentinel.
            raise TypeError("Implicit conversion to a NumPy array is not allowed.")

    class _BaseGetFock:
        def get_fock(self, *, h1e=None, **_kwargs):
            return h1e

    class DummyCDFT(_CDFTMixin, _BaseGetFock):
        pass

    system = ConstraintSystem(
        names=["q_tot"],
        kinds=["net_charge"],
        targets=np.asarray([0.0], dtype=float),
        operators=[np.eye(2, dtype=float)],
        report_scales=np.asarray([-1.0], dtype=float),
        report_offsets=np.asarray([0.0], dtype=float),
    )
    mf = DummyCDFT()
    mf._init_cdft(constraint_system=system, initial_vc=[0.0])

    h1e = FakeGPUArray(np.eye(2, dtype=float))
    vhf = FakeGPUArray(np.zeros((2, 2, 2), dtype=float))
    s1e = FakeGPUArray(np.eye(2, dtype=float))
    dm = FakeGPUArray(np.zeros((2, 2, 2), dtype=float))

    fock = mf.get_fock(h1e=h1e, s1e=s1e, vhf=vhf, dm=dm, cycle=-1)
    getter = getattr(fock, "get", None)
    fock_np = getter() if callable(getter) else np.asarray(fock)
    assert fock_np.shape == (2, 2)


def test_update_vc_rejects_non_improving_failed_root(monkeypatch) -> None:
    """Failed inner solves should not overwrite vc with a worse candidate."""

    class DummyCDFT(_CDFTMixin):
        def _density_from_fock(self: Any, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
            _ = overlap
            return np.asarray(fock, dtype=float)

    def fake_root(_objective, x0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray([10.0], dtype=float),
            success=False,
            status=2,
            nfev=1,
        )

    monkeypatch.setattr(meanfield.optimize, "root", fake_root)

    system = ConstraintSystem(
        names=["q_tot"],
        kinds=["net_charge"],
        targets=np.asarray([0.0], dtype=float),
        operators=[np.eye(2, dtype=float)],
        report_scales=np.asarray([-1.0], dtype=float),
        report_offsets=np.asarray([0.0], dtype=float),
    )
    mf = DummyCDFT()
    mf._init_cdft(constraint_system=system, initial_vc=[UNCHANGED_VC], vc_max_step=0.25)
    base_fock = np.zeros((2, 2, 2), dtype=float)
    overlap = np.eye(2, dtype=float)

    mf._update_vc(base_fock=base_fock, overlap=overlap)
    assert mf.vc[0] == UNCHANGED_VC


def test_update_vc_applies_bounded_step_on_success(monkeypatch) -> None:
    """Even successful roots should move vc in bounded residual-improving steps."""

    class DummyCDFT(_CDFTMixin):
        def _density_from_fock(self: Any, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
            _ = overlap
            return np.asarray(fock, dtype=float)

    def fake_root(_objective, x0, **_kwargs):
        return SimpleNamespace(
            x=np.asarray([0.0], dtype=float),
            success=True,
            status=1,
            nfev=1,
        )

    monkeypatch.setattr(meanfield.optimize, "root", fake_root)

    system = ConstraintSystem(
        names=["q_tot"],
        kinds=["net_charge"],
        targets=np.asarray([0.0], dtype=float),
        operators=[np.eye(2, dtype=float)],
        report_scales=np.asarray([-1.0], dtype=float),
        report_offsets=np.asarray([0.0], dtype=float),
    )
    mf = DummyCDFT()
    mf._init_cdft(constraint_system=system, initial_vc=[1.0], vc_max_step=0.25)
    base_fock = np.zeros((2, 2, 2), dtype=float)
    overlap = np.eye(2, dtype=float)

    mf._update_vc(base_fock=base_fock, overlap=overlap)
    assert mf.vc[0] == BOUNDED_VC
