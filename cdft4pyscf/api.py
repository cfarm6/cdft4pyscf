"""Public constructors for class-based constrained DFT workflows."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pyscf import gto

from cdft4pyscf.meanfield import CDFT_UKS, CDFT_UKS_GPU

if TYPE_CHECKING:
    from cdft4pyscf.models import RunRequest


def build_cdft_mean_field(request: "RunRequest") -> Any:
    """Build a constrained UKS mean-field object from a validated request."""
    mol = gto.M(
        atom=request.atom,
        basis=request.basis,
        spin=request.spin,
        charge=request.charge,
        verbose=request.options.verbosity,
    )

    if request.backend == "cpu":
        mf = CDFT_UKS(
            mol,
            constraints=request.constraints,
            population_basis=request.population.basis,
            initial_vc=request.options.initial_vc,
            conv_tol=request.options.conv_tol,
            vc_tol=request.options.vc_tol,
            vc_max_cycle=request.options.vc_max_cycle,
            vc_max_step=request.options.vc_max_step,
            log_inner_solver=request.options.log_inner_solver,
        )
    elif request.backend == "gpu":
        mf = CDFT_UKS_GPU(
            mol,
            constraints=request.constraints,
            population_basis=request.population.basis,
            initial_vc=request.options.initial_vc,
            conv_tol=request.options.conv_tol,
            vc_tol=request.options.vc_tol,
            vc_max_cycle=request.options.vc_max_cycle,
            vc_max_step=request.options.vc_max_step,
            log_inner_solver=request.options.log_inner_solver,
        )
    else:
        msg = f"Unsupported backend '{request.backend}'."
        raise ValueError(msg)

    mf.max_cycle = request.options.max_cycle
    mf.conv_tol = float(request.options.energy_tol)
    mf.xc = request.xc
    mf.verbose = request.options.verbosity
    return mf
