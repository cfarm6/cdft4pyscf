# API Overview

Use these pages depending on what you need:

- [API by Workflow](api-workflow.md): start with task-oriented steps from
  defining regions and constraints through inspecting results.
- [API Reference](api-reference.md): complete function/model/exception reference.

## Public symbols exported by `cdft4pyscf`

### Class-based mean-field API

- `CDFT_UKS`
- `build_cdft_mean_field`

### Models

- `RegionSpec`
- `ConstraintSpec`
- `SolverOptions`
- `RunRequest`

### Exceptions

- `CdftError`
- `ConvergenceError`
- `BackendUnavailableError`
