# API Reference

This page documents the full public API exported by `cdft4pyscf`.
Where available, descriptions are taken directly from code docstrings.

## Module overview

- Module docstring: **"Public constructors for class-based constrained DFT workflows."**

## Class and constructor reference

### `CDFT_UKS(mol, *, constraints, **kwargs)`

Docstring: **"Class-first constrained UKS object for PySCF CPU workflows."**

#### What it does

1. Builds a constrained UKS mean-field object from a `pyscf.gto.Mole`.
2. Injects cDFT multiplier updates into the `kernel()`/`get_fock()` lifecycle.
3. Exposes post-SCF helper methods for constraint values, residuals, and multipliers.

#### Inputs

- `mol`: PySCF molecule object.
- `constraints`: list of `ConstraintSpec`.
- optional kwargs:
  - `initial_vc`
  - `conv_tol` (cDFT residual tolerance)
  - `vc_tol` and `vc_max_cycle` (inner multiplier solve controls)
  - `log_inner_solver`
  - `raise_on_unconverged`

#### Important methods/attributes

- `kernel()` returns total energy and runs constrained SCF.
- `constraint_values()` returns `dict[str, float]`.
- `constraint_residuals()` returns `dict[str, float]`.
- `multiplier_by_constraint()` returns `dict[str, float]`.
- `vc` stores the latest multiplier vector.
- `cdft_residual_norm` and `cdft_inner_calls` store diagnostics.

#### Raised exceptions

- `ConvergenceError` if SCF/cDFT convergence criteria are not met and
  `raise_on_unconverged=True`.

### `build_cdft_mean_field(request: RunRequest) -> Any`

Builds a configured constrained mean-field object from a typed request.

- `backend="cpu"` returns `CDFT_UKS`.
- `backend="gpu"` returns a GPU4PySCF-backed constrained UKS class, or raises
  `BackendUnavailableError` when unavailable.

## Models

Module docstring: **"Typed request/response models for cDFT workflows."**

### `RegionSpec`

Docstring: **"Region definition supported by v0.1."**

Defines a named region as an atom-index list.

- `name: str` - region identifier used by constraints.
- `atom_indices: list[int]` - non-empty list of atom indices.

Validation behavior:

- atom indices must be unique.
- atom indices must be non-negative.

### `ConstraintSpec`

Docstring: **"Constraint specification for cDFT."**

Defines one constraint equation and its target.

- `name: str` - unique constraint identifier.
- `kind: Literal["electron_number", "net_charge"]` - constraint type.
- `target: float` - target value for the chosen constraint.
- `region: RegionSpec | list[RegionSpec] | None` - required for
  `electron_number`; required and single `RegionSpec` for `net_charge`.

Validation behavior:

- `electron_number` constraints require `region`.
- `net_charge` constraints require `region`.
- `net_charge` constraints require a single `RegionSpec` (lists are rejected).

### `SolverOptions`

Docstring: **"Numerical controls for the outer SCF and inner multiplier solves."**

- `max_cycle: int = 50` - max outer iterations (`>= 1`).
- `conv_tol: float = 1e-7` - residual tolerance (`> 0`).
- `dm_tol: float = 1e-6` - density-matrix change tolerance (`> 0`).
- `vc_tol: float = 1e-6` - inner multiplier solve tolerance (`> 0`).
- `vc_max_cycle: int = 50` - max inner iterations (`>= 1`).
- `verbosity: int = 0` - logging level (`0..3`).
- `log_inner_solver: bool = False` - include per-iteration inner-solver logs.

### `RunRequest`

Docstring: **"Top-level request payload for a cDFT run."**

Run configuration object passed into `build_cdft_mean_field`.

- `atom: str` - required geometry text.
- `basis: str = "sto-3g"`
- `xc: str = "lda,vwn"`
- `spin: int = 0`
- `charge: int = 0`
- `backend: Literal["cpu", "gpu"] = "cpu"`
- `regions: list[RegionSpec] = []`
- `constraints: list[ConstraintSpec]` - required, must contain at least one
  item.
- `options: SolverOptions = SolverOptions()`

Validation behavior:

- region names must be unique.
- constraint names must be unique.
- constraints referencing a region must reference a defined region name.

## Exceptions

### `CdftError`

Docstring: **"Base package exception."**

Base exception carrying a human-readable `message`.

### `ConvergenceError`

Docstring: **"Raised when the solver fails to satisfy convergence criteria."**

Extends `CdftError` and can include `diagnostics: list[str]` that are appended
to the error message string.

### `BackendUnavailableError`

Docstring: **"Raised when a requested backend cannot be initialized."**

## Typical import pattern

```python
from cdft4pyscf import (
    BackendUnavailableError,
    CDFT_UKS,
    CdftError,
    ConstraintSpec,
    ConvergenceError,
    RegionSpec,
    RunRequest,
    SolverOptions,
    build_cdft_mean_field,
)
```
