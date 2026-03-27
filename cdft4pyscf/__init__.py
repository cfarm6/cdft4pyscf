"""cdft4pyscf package."""

from cdft4pyscf.api import build_cdft_mean_field
from cdft4pyscf.exceptions import BackendUnavailableError, CdftError, ConvergenceError
from cdft4pyscf.meanfield import CDFT
from cdft4pyscf.models import (
    Constraint,
    FragmentTerm,
    ProjectorSpec,
    RunRequest,
    SolverOptions,
)

__all__ = [
    "CDFT",
    "BackendUnavailableError",
    "CdftError",
    "Constraint",
    "ConvergenceError",
    "FragmentTerm",
    "ProjectorSpec",
    "RunRequest",
    "SolverOptions",
    "build_cdft_mean_field",
]
