# API by Workflow

Use this path when building a cDFT run from scratch.

## 1) Define constrained regions

Create one or more [`RegionSpec`](api-reference.md#regionspec) values.

- Set `name` to a unique region identifier.
- Set `atom_indices` to non-negative, unique atom indices.

## 2) Define constraints

Create one or more [`ConstraintSpec`](api-reference.md#constraintspec) values.

- Use `kind="electron_number"` with a required `region`.
- Use `kind="net_charge"` with a required single `region`.
- Give each constraint a unique `name`.

## 3) Choose solver controls (optional)

Customize [`SolverOptions`](api-reference.md#solveroptions) if defaults are not
sufficient.

- Outer-loop controls: `max_cycle`, `scf_conv_tol`, `constraint_conv_tol`.
- Inner-loop controls: `inner_vc_max_cycle`, `inner_vc_tol`, `inner_vc_max_step`.
- Logging controls: `verbosity`, `inner_solver_trace`.

## 4) Build a constrained mean-field object

Preferred class-first flow:

- Build a PySCF molecule (`gto.M`).
- Construct [`CDFT_UKS`](api-reference.md#cdft_uksmol-constraints-kwargs).
- Set normal MF settings (`mf.xc`, `mf.max_cycle`, grid controls, etc.).

Alternative helper flow:

- Build a [`RunRequest`](api-reference.md#runrequest) with molecule settings.
- Call [`build_cdft_mean_field(request)`](api-reference.md#build_cdft_mean_fieldrequest).

## 5) Run cDFT

Call `mf.kernel()` on your constrained mean-field object.

- Returns a total energy as in standard PySCF.
- May raise [`ConvergenceError`](api-reference.md#convergenceerror) when SCF or
  cDFT residual criteria are not satisfied.

## 6) Inspect outputs and diagnostics

Inspect cDFT helper methods on the mean-field object:

- `multiplier_by_constraint()` for optimized multipliers.
- `constraint_values()` for final measured values.
- `constraint_residuals()` for final residuals.
- `cdft_residual_norm` and `cdft_inner_calls` for diagnostics.
- `cdft_diagnostics()` for a single explicit diagnostics snapshot.
