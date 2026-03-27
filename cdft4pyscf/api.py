"""Public constructors for wrapper-based constrained DFT workflows."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

from pyscf import dft, gto

from cdft4pyscf.exceptions import BackendUnavailableError
from cdft4pyscf.meanfield import CDFT

if TYPE_CHECKING:
    from cdft4pyscf.models import RunRequest


def build_cdft_mean_field(request: "RunRequest") -> CDFT:
    """Build a constrained wrapper object from a validated request."""
    mol = gto.M(
        atom=request.atom,
        basis=request.basis,
        spin=request.spin,
        charge=request.charge,
        verbose=request.options.verbosity,
    )

    if request.backend == "cpu":
        base = dft.UKS(mol)
    elif request.backend == "gpu":
        try:
            gpu_uks = importlib.import_module("gpu4pyscf.dft.uks")
        except Exception as exc:
            msg = "GPU backend requested but gpu4pyscf could not be imported."
            raise BackendUnavailableError(msg) from exc
        base = gpu_uks.UKS(mol)
    else:
        msg = f"Unsupported backend '{request.backend}'."
        raise ValueError(msg)

    base.max_cycle = request.options.max_cycle
    base.conv_tol = float(request.options.scf_conv_tol)
    base.xc = request.xc
    base.verbose = request.options.verbosity
    return CDFT(base, constraints=request.constraints, solver=request.options)
