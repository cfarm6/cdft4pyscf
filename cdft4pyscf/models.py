"""Typed request/response models for cDFT workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ConstraintKind = Literal["electron_number", "net_charge"]
BackendKind = Literal["cpu", "gpu"]
PopulationBasis = Literal["lowdin", "meta_lowdin", "iao", "nao"]
MoLocalizationMethod = Literal["boys", "pipek", "ibo", "edmiston", "cholesky"]
MoLocalizationSpace = Literal["occ", "vir", "all"]
MoLocalizationSpin = Literal["alpha", "beta", "both"]


class RegionSpec(BaseModel):
    """Region definition supported by v0.1."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    atom_indices: list[int] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_atom_indices(self) -> "RegionSpec":
        if len(set(self.atom_indices)) != len(self.atom_indices):
            msg = "Region atom_indices must be unique."
            raise ValueError(msg)
        if any(index < 0 for index in self.atom_indices):
            msg = "Region atom_indices must be non-negative."
            raise ValueError(msg)
        return self


class ConstraintSpec(BaseModel):
    """Constraint specification for cDFT."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    kind: ConstraintKind
    target: float
    region: RegionSpec | list[RegionSpec] | None = None

    @model_validator(mode="after")
    def _validate_region_requirement(self) -> "ConstraintSpec":
        if self.kind == "electron_number" and self.region is None:
            msg = "electron_number constraints require a region."
            raise ValueError(msg)
        if (
            self.kind == "electron_number"
            and isinstance(self.region, list)
            and len(self.region) == 0
        ):
            msg = "electron_number constraints require at least one region."
            raise ValueError(msg)
        if self.kind == "net_charge" and self.region is None:
            msg = "net_charge constraints require a region."
            raise ValueError(msg)
        if self.kind == "net_charge" and isinstance(self.region, list):
            msg = "net_charge constraints require a single region."
            raise ValueError(msg)
        return self


class SolverOptions(BaseModel):
    """Numerical controls for the outer SCF and inner multiplier solves."""

    model_config = ConfigDict(extra="forbid")

    max_cycle: int = Field(default=50, ge=1)
    conv_tol: float = Field(default=1e-7, gt=0.0)
    dm_tol: float = Field(default=1e-6, gt=0.0)
    vc_tol: float = Field(default=1e-6, gt=0.0)
    vc_max_cycle: int = Field(default=50, ge=1)
    vc_max_step: float = Field(default=0.25, ge=0.0)
    initial_vc: list[float] | None = None
    energy_tol: float = Field(default=1e-6, gt=0.0)
    verbosity: int = Field(default=0, ge=0, le=3)
    log_inner_solver: bool = False


class MoLocalizationOptions(BaseModel):
    """Optional post-run MO localization configuration (analysis only)."""

    model_config = ConfigDict(extra="forbid")

    method: MoLocalizationMethod
    space: MoLocalizationSpace = "occ"
    spin: MoLocalizationSpin = "both"


class PopulationOptions(BaseModel):
    """Population analysis settings used to build constraint operators."""

    model_config = ConfigDict(extra="forbid")

    basis: PopulationBasis = "lowdin"
    localize_mos: MoLocalizationOptions | None = None


class RunRequest(BaseModel):
    """Top-level request payload for a cDFT run."""

    model_config = ConfigDict(extra="forbid")

    atom: str = Field(min_length=1)
    basis: str = Field(default="sto-3g", min_length=1)
    xc: str = Field(default="lda,vwn", min_length=1)
    spin: int = Field(default=0, ge=0)
    charge: int = 0
    backend: BackendKind = "cpu"
    regions: list[RegionSpec] = Field(default_factory=list)
    constraints: list[ConstraintSpec] = Field(min_length=1)
    options: SolverOptions = Field(default_factory=SolverOptions)
    population: PopulationOptions = Field(default_factory=PopulationOptions)

    @model_validator(mode="after")
    def _validate_unique_names(self) -> "RunRequest":
        region_names = [region.name for region in self.regions]
        if len(set(region_names)) != len(region_names):
            msg = "Region names must be unique."
            raise ValueError(msg)
        constraint_names = [constraint.name for constraint in self.constraints]
        if len(set(constraint_names)) != len(constraint_names):
            msg = "Constraint names must be unique."
            raise ValueError(msg)
        return self


class RunDiagnostics(BaseModel):
    """Structured diagnostics emitted by the solver."""

    model_config = ConfigDict(extra="forbid")

    outer_iterations: int
    inner_iterations: int
    residual_norm: float
    dm_delta_norm: float
    messages: list[str] = Field(default_factory=list)


class RunResult(BaseModel):
    """Top-level result payload for a cDFT run."""

    model_config = ConfigDict(extra="forbid")

    converged: bool
    backend: BackendKind
    total_energy: float
    vc_by_constraint: dict[str, float]
    value_by_constraint: dict[str, float]
    diagnostics: RunDiagnostics
