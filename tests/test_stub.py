"""API-level smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.models import ConstraintSpec, RegionSpec, RunRequest


def test_build_cdft_mean_field_cpu_smoke(monkeypatch) -> None:
    """CPU constructor should return a constrained mean-field object."""
    request = RunRequest(
        atom="H 0 0 0; H 0 0 0.74",
        constraints=[
            ConstraintSpec(
                name="n_frag_a",
                kind="electron_number",
                target=1.0,
                region=RegionSpec(name="frag_a", atom_indices=[0]),
            )
        ],
    )

    fake_mol = SimpleNamespace(
        nelectron=2,
        aoslice_by_atom=lambda: [[0, 0, 0, 1], [0, 0, 1, 2]],
    )
    monkeypatch.setattr("cdft4pyscf.api.gto.M", lambda **_kwargs: fake_mol)

    class FakeCDFT:
        def __init__(self, mol, **kwargs):
            self.mol = mol
            self.kwargs = kwargs
            self.xc = ""
            self.verbose = 0
            self.max_cycle = 0
            self.conv_tol = 0.0

    monkeypatch.setattr("cdft4pyscf.api.CDFT_UKS", FakeCDFT)
    mf = build_cdft_mean_field(request)
    assert mf.kwargs["constraints"][0].name == "n_frag_a"
    assert mf.xc == request.xc


def test_build_cdft_mean_field_gpu_uses_cdft_uks_gpu(monkeypatch) -> None:
    """GPU constructor should route through CDFT_UKS_GPU class."""
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

    fake_mol = SimpleNamespace(
        nelectron=2,
        aoslice_by_atom=lambda: [[0, 0, 0, 1], [0, 0, 1, 2]],
        atom_charges=lambda: [1.0, 1.0],
    )
    monkeypatch.setattr("cdft4pyscf.api.gto.M", lambda **_kwargs: fake_mol)

    class FakeGPUConstructor:
        def __init__(self, mol, **kwargs):
            self.mol = mol
            self.kwargs = kwargs
            self.xc = ""
            self.verbose = 0
            self.max_cycle = 0
            self.conv_tol = 0.0

    monkeypatch.setattr("cdft4pyscf.api.CDFT_UKS_GPU", FakeGPUConstructor)

    mf = build_cdft_mean_field(request)
    assert mf.kwargs["constraints"][0].name == "q_tot"
    assert mf.xc == request.xc
