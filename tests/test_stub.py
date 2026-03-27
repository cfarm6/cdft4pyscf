"""API-level smoke tests."""

from __future__ import annotations

from types import SimpleNamespace

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.meanfield import CDFT
from cdft4pyscf.models import Constraint, FragmentTerm, RunRequest


def test_build_cdft_mean_field_cpu_smoke(monkeypatch) -> None:
    """CPU constructor should return a wrapper constrained mean-field object."""
    request = RunRequest(
        atom="H 0 0 0; H 0 0 0.74",
        constraints=[
            Constraint(
                name="n_frag_a",
                fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                target=1.0,
            )
        ],
    )

    fake_mol = SimpleNamespace(
        nelectron=2,
        atom_charges=lambda: [1.0, 1.0],
        aoslice_by_atom=lambda: [[0, 0, 0, 1], [0, 0, 1, 2]],
    )
    monkeypatch.setattr("cdft4pyscf.api.gto.M", lambda **_kwargs: fake_mol)

    class FakeBaseMF:
        def __init__(self, mol):
            self.mol = mol
            self.xc = ""
            self.verbose = 0
            self.max_cycle = 0
            self.conv_tol = 0.0

        def get_ovlp(self, _mol):
            return [[1.0, 0.0], [0.0, 1.0]]

        def get_fock(self, **_kwargs):
            return [[1.0]]

    monkeypatch.setattr("cdft4pyscf.api.dft.UKS", FakeBaseMF)
    mf = build_cdft_mean_field(request)
    assert isinstance(mf, CDFT)
    assert mf.constraints[0].name == "n_frag_a"
    assert mf.xc == request.xc
    assert mf.conv_tol == request.options.scf_conv_tol


def test_build_cdft_mean_field_gpu_uses_cdft_uks_gpu(monkeypatch) -> None:
    """GPU constructor should route through gpu4pyscf then wrap with CDFT."""
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

    fake_mol = SimpleNamespace(
        nelectron=2,
        aoslice_by_atom=lambda: [[0, 0, 0, 1], [0, 0, 1, 2]],
        atom_charges=lambda: [1.0, 1.0],
    )
    monkeypatch.setattr("cdft4pyscf.api.gto.M", lambda **_kwargs: fake_mol)

    class FakeGPUConstructor:
        def __init__(self, mol):
            self.mol = mol
            self.xc = ""
            self.verbose = 0
            self.max_cycle = 0
            self.conv_tol = 0.0

        def get_ovlp(self, _mol):
            return [[1.0, 0.0], [0.0, 1.0]]

        def get_fock(self, **_kwargs):
            return [[1.0]]

    class FakeGPUUKSModule:
        UKS = FakeGPUConstructor

    monkeypatch.setattr("cdft4pyscf.api.importlib.import_module", lambda _name: FakeGPUUKSModule)

    mf = build_cdft_mean_field(request)
    assert isinstance(mf, CDFT)
    assert mf.constraints[0].name == "q_tot"
    assert mf.xc == request.xc
    assert mf.conv_tol == request.options.scf_conv_tol
