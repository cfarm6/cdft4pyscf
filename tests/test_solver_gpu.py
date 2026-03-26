"""GPU backend tests."""

from __future__ import annotations

import numpy as np

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.constraints import ConstraintSystem
from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT_UKS_GPU, _CDFTMixin
from cdft4pyscf.models import ConstraintSpec, RegionSpec, RunRequest


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
