"""Typed models for the wrapper-first cDFT API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

BackendKind = Literal["cpu", "gpu"]
TargetType = Literal["electrons", "charge"]
SpinMode = Literal["total", "spin"]
ProjectorMethod = Literal[
    "mulliken",
    "lowdin",
    "minao",
    "iao",
    "lo:boys",
    "lo:pm",
    "lo:er",
    "becke",
]
SolverMode = Literal["micro", "outer_newton", "newton_kkt", "penalty"]
FallbackMode = Literal["micro", "outer_newton", "newton_kkt", "penalty"]


class FragmentTerm(BaseModel):
    """A linear term in a constraint equation."""

    model_config = ConfigDict(extra="forbid")

    atoms: list[int] = Field(min_length=1)
    coeff: float = 1.0
    orbital_selector: str | None = None

    @model_validator(mode="after")
    def _validate_atoms(self) -> "FragmentTerm":
        if len(set(self.atoms)) != len(self.atoms):
            msg = "FragmentTerm atoms must be unique."
            raise ValueError(msg)
        if any(atom < 0 for atom in self.atoms):
            msg = "FragmentTerm atoms must be non-negative."
            raise ValueError(msg)
        if self.coeff == 0.0:
            msg = "FragmentTerm coeff cannot be zero."
            raise ValueError(msg)
        return self


class ProjectorSpec(BaseModel):
    """Projector backend and optional backend-specific parameters."""

    model_config = ConfigDict(extra="forbid")

    method: ProjectorMethod = "lowdin"
    params: dict[str, str | int | float | bool] = Field(default_factory=dict)
    cache: bool = True


class Constraint(BaseModel):
    """Public cDFT constraint equation."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    fragments: list[FragmentTerm] = Field(min_length=1)
    target: float
    target_type: TargetType = "electrons"
    spin_mode: SpinMode = "total"
    projector: ProjectorSpec = Field(default_factory=ProjectorSpec)

    @model_validator(mode="after")
    def _validate_fragments(self) -> "Constraint":
        return self


class SolverOptions(BaseModel):
    """Numerical controls and mode selection for cDFT enforcement."""

    model_config = ConfigDict(extra="forbid")

    mode: SolverMode = "micro"
    fallback_modes: list[FallbackMode] = Field(
        default_factory=lambda: ["micro", "outer_newton", "newton_kkt", "penalty"]
    )
    max_cycle: int = Field(default=50, ge=1)
    scf_conv_tol: float = Field(default=1e-6, gt=0.0)
    conv_tol_constraint: float = Field(default=1e-6, gt=0.0)
    inner_vc_tol: float = Field(default=1e-6, gt=0.0)
    inner_vc_max_cycle: int = Field(default=50, ge=1)
    max_v_step: float = Field(default=0.25, ge=0.0)
    damping: float = Field(default=0.5, gt=0.0, le=1.0)
    penalty_lambda: float = Field(default=100.0, gt=0.0)
    initial_v_lagrange: list[float] | None = None
    verbosity: int = Field(default=0, ge=0, le=5)
    trace: bool = False


class RunRequest(BaseModel):
    """Top-level request payload for a cDFT run."""

    model_config = ConfigDict(extra="forbid")

    atom: str = Field(min_length=1)
    basis: str = Field(default="sto-3g", min_length=1)
    xc: str = Field(default="lda,vwn", min_length=1)
    spin: int = Field(default=0, ge=0)
    charge: int = 0
    backend: BackendKind = "cpu"
    constraints: list[Constraint] = Field(min_length=1)
    options: SolverOptions = Field(default_factory=SolverOptions)

    @model_validator(mode="after")
    def _validate_unique_names(self) -> "RunRequest":
        names = [constraint.name for constraint in self.constraints]
        if len(set(names)) != len(names):
            msg = "Constraint names must be unique."
            raise ValueError(msg)
        return self
