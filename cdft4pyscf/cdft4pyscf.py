"""Compatibility exports for the cdft4pyscf package."""

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.meanfield import CDFT_UKS, CDFT_UKS_GPU
from cdft4pyscf.models import ConstraintSpec, RegionSpec, RunRequest, RunResult, SolverOptions

__all__ = [
    "CDFT_UKS",
    "CDFT_UKS_GPU",
    "ConstraintSpec",
    "RegionSpec",
    "RunRequest",
    "RunResult",
    "SolverOptions",
    "build_cdft_mean_field",
]
