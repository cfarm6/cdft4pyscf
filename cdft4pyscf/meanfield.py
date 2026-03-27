"""Wrapper-first constrained mean-field objects."""

from __future__ import annotations

import importlib
import logging
from types import MethodType
from typing import Any

import numpy as np
from scipy import optimize
from scipy.sparse.linalg import minres

from cdft4pyscf.constraints import (
    build_constraint_system,
    evaluate_constraint_residuals,
    evaluate_constraint_values,
    report_constraint_residuals,
    report_constraint_values,
)
from cdft4pyscf.exceptions import ConvergenceError
from cdft4pyscf.models import Constraint, SolverOptions

LOGGER = logging.getLogger(__name__)


def _cupy_module() -> Any | None:
    """Load CuPy lazily when available."""
    try:
        return importlib.import_module("cupy")
    except Exception:
        return None


def _is_cupy_array(value: Any) -> bool:
    """Return true when value is a CuPy ndarray."""
    cp = _cupy_module()
    if cp is None:
        return False
    return isinstance(value, cp.ndarray)


def _to_numpy_array(value: Any) -> np.ndarray:
    """Convert NumPy/CuPy-like arrays to a NumPy ndarray."""
    if isinstance(value, np.ndarray):
        return value
    getter = getattr(value, "get", None)
    if callable(getter):
        return np.asarray(getter())
    return np.asarray(value)


def _to_backend_array(reference: Any, value: np.ndarray) -> Any:
    """Convert NumPy value to the backend of reference array."""
    if _is_cupy_array(reference):
        cp = _cupy_module()
        if cp is not None:
            return cp.asarray(value)
    return np.asarray(value)


def _stack_spin_matrices(alpha: Any, beta: Any) -> Any:
    """Stack alpha and beta blocks preserving backend arrays."""
    if _is_cupy_array(alpha) and _is_cupy_array(beta):
        cp = _cupy_module()
        if cp is not None:
            return cp.stack([alpha, beta])
    return np.stack([_to_numpy_array(alpha), _to_numpy_array(beta)])


class CDFT:
    """Wrapper-first cDFT interface around an existing mean-field object."""

    def __init__(
        self,
        mf: Any,
        *,
        constraints: list[Constraint],
        solver: SolverOptions | None = None,
    ) -> None:
        object.__setattr__(self, "mf", mf)
        object.__setattr__(self, "constraints", constraints)
        object.__setattr__(self, "solver_options", solver or SolverOptions())

        overlap = _to_numpy_array(mf.get_ovlp(mf.mol)).astype(float, copy=False)
        system = build_constraint_system(
            constraints=constraints,
            mol=mf.mol,
            ao_slices=mf.mol.aoslice_by_atom(),
            atom_charges=np.asarray(mf.mol.atom_charges(), dtype=float),
            overlap=overlap,
        )
        object.__setattr__(self, "constraint_system", system)

        n_constraints = len(system.names)
        initial = self.solver_options.initial_v_lagrange
        if initial is None:
            initial_v = np.zeros(n_constraints, dtype=float)
        else:
            initial_v = np.asarray(initial, dtype=float)
            if initial_v.shape != (n_constraints,):
                msg = (
                    "initial_v_lagrange length must match number of constraints "
                    f"({n_constraints}); got {int(initial_v.size)}"
                )
                raise ValueError(msg)
        object.__setattr__(self, "v_lagrange", initial_v)
        object.__setattr__(self, "converged", False)
        object.__setattr__(
            self,
            "solver_state",
            {
                "mode": self.solver_options.mode,
                "fallback_used": None,
                "inner_solver_evaluations": 0,
                "residual_norm": float("inf"),
                "refinement_steps": 0,
                "trace_messages": [],
            },
        )
        object.__setattr__(self, "_last_residuals", np.zeros(n_constraints, dtype=float))
        object.__setattr__(self, "_orig_get_fock", mf.get_fock)
        mf.get_fock = MethodType(CDFT._patched_get_fock, self)

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped mean-field object."""
        return getattr(self.mf, name)

    def __setattr__(self, name: str, value: Any) -> None:
        """Store wrapper attributes locally and delegate everything else."""
        own = {
            "mf",
            "constraints",
            "solver_options",
            "constraint_system",
            "v_lagrange",
            "converged",
            "solver_state",
            "_last_residuals",
            "_orig_get_fock",
        }
        if name in own:
            object.__setattr__(self, name, value)
        else:
            setattr(self.mf, name, value)

    def build_projectors(self) -> list[np.ndarray]:
        """Return dense AO projectors for all constraints."""
        return [operator.as_dense() for operator in self.constraint_system.operators]

    def _constraint_operator(self, multipliers: np.ndarray) -> np.ndarray:
        dense = np.zeros_like(self.constraint_system.operators[0].as_dense())
        for multiplier, operator in zip(multipliers, self.constraint_system.operators, strict=True):
            dense += float(multiplier) * operator.as_dense()
        return dense

    def _density_from_fock(self, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        mo_energy, mo_coeff = self.mf.eig(fock, overlap)
        mo_occ = self.mf.get_occ(mo_energy, mo_coeff)
        return _to_numpy_array(self.mf.make_rdm1(mo_coeff, mo_occ)).astype(float, copy=False)

    def _evaluate_residual_for_multipliers(
        self, multipliers: np.ndarray, base_fock: np.ndarray, overlap: np.ndarray
    ) -> np.ndarray:
        operator = self._constraint_operator(multipliers)
        operator_like = _to_backend_array(base_fock[0], operator)
        constrained_fock = _stack_spin_matrices(
            base_fock[0] + operator_like,
            base_fock[1] + operator_like,
        )
        dm = self._density_from_fock(constrained_fock, overlap)
        values = evaluate_constraint_values(dm[0] + dm[1], self.constraint_system)
        return values - self.constraint_system.targets

    def _micro_step(self, base_fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        result = optimize.root(
            lambda v: self._evaluate_residual_for_multipliers(v, base_fock, overlap),
            self.v_lagrange,
            method="hybr",
            options={
                "xtol": self.solver_options.inner_vc_tol,
                "maxfev": self.solver_options.inner_vc_max_cycle,
            },
        )
        self.solver_state["inner_solver_evaluations"] += int(result.nfev)
        delta = np.asarray(result.x, dtype=float) - self.v_lagrange
        step = float(self.solver_options.max_v_step)
        if step > 0.0:
            delta = np.clip(delta, -step, step)
        return self.v_lagrange + delta

    def _outer_newton_step(self, base_fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        v0 = np.asarray(self.v_lagrange, dtype=float)
        r0 = self._evaluate_residual_for_multipliers(v0, base_fock, overlap)
        n_constraints = len(v0)
        jac = np.zeros((n_constraints, n_constraints), dtype=float)
        eps = 1e-3
        for index in range(n_constraints):
            probe = v0.copy()
            probe[index] += eps
            jac[:, index] = (
                self._evaluate_residual_for_multipliers(probe, base_fock, overlap) - r0
            ) / eps
        try:
            delta = np.linalg.solve(jac, -r0)
        except np.linalg.LinAlgError:
            delta, *_ = np.linalg.lstsq(jac, -r0, rcond=None)
        delta *= float(self.solver_options.damping)
        step = float(self.solver_options.max_v_step)
        if step > 0.0:
            delta = np.clip(delta, -step, step)
        return v0 + delta

    def _newton_kkt_step(self, base_fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        v0 = np.asarray(self.v_lagrange, dtype=float)
        r0 = self._evaluate_residual_for_multipliers(v0, base_fock, overlap)
        n_constraints = len(v0)
        jac = np.zeros((n_constraints, n_constraints), dtype=float)
        eps = 1e-3
        for index in range(n_constraints):
            probe = v0.copy()
            probe[index] += eps
            jac[:, index] = (
                self._evaluate_residual_for_multipliers(probe, base_fock, overlap) - r0
            ) / eps
        kkt = jac.T @ jac + np.eye(n_constraints, dtype=float) * 1e-8
        rhs = -jac.T @ r0
        delta, _ = minres(kkt, rhs, rtol=self.solver_options.inner_vc_tol, maxiter=50)
        trust_radius = float(self.solver_options.max_v_step) * max(
            1.0, np.sqrt(float(n_constraints))
        )
        norm = float(np.linalg.norm(delta))
        if norm > trust_radius and norm > 0.0:
            delta *= trust_radius / norm
        return v0 + delta

    def _penalty_step(self, base_fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        residuals = self._evaluate_residual_for_multipliers(self.v_lagrange, base_fock, overlap)
        update = -2.0 * float(self.solver_options.penalty_lambda) * residuals
        update *= float(self.solver_options.damping)
        step = float(self.solver_options.max_v_step)
        if step > 0.0:
            update = np.clip(update, -step, step)
        return self.v_lagrange + update

    def _solve_multipliers(self, base_fock: np.ndarray, overlap: np.ndarray) -> None:
        modes = [self.solver_options.mode, *self.solver_options.fallback_modes]
        best_v = np.asarray(self.v_lagrange, dtype=float)
        best_r = self._evaluate_residual_for_multipliers(best_v, base_fock, overlap)
        best_norm = float(np.linalg.norm(best_r))
        used_mode = None

        for mode in modes:
            if mode == "micro":
                candidate = self._micro_step(base_fock, overlap)
            elif mode == "outer_newton":
                candidate = self._outer_newton_step(base_fock, overlap)
            elif mode == "newton_kkt":
                candidate = self._newton_kkt_step(base_fock, overlap)
            elif mode == "penalty":
                candidate = self._penalty_step(base_fock, overlap)
            else:
                continue
            residuals = self._evaluate_residual_for_multipliers(candidate, base_fock, overlap)
            norm = float(np.linalg.norm(residuals))
            if norm <= best_norm:
                best_v = np.asarray(candidate, dtype=float)
                best_r = np.asarray(residuals, dtype=float)
                best_norm = norm
                used_mode = mode
            if best_norm <= self.solver_options.conv_tol_constraint:
                break

        self.v_lagrange = best_v
        self._last_residuals = best_r
        self.solver_state["mode"] = self.solver_options.mode
        self.solver_state["fallback_used"] = used_mode
        self.solver_state["residual_norm"] = best_norm
        if self.solver_options.trace:
            msg = (
                f"[cDFT] multiplier solve mode={used_mode} residual_norm={best_norm:.3e} "
                f"v_lagrange={np.array2string(self.v_lagrange, precision=6, separator=', ')}"
            )
            LOGGER.info(msg)
            self.solver_state["trace_messages"].append(msg)

    def _patched_get_fock(
        self,
        h1e: Any | None = None,
        s1e: Any | None = None,
        vhf: Any | None = None,
        dm: Any | None = None,
        cycle: int = -1,
        diis: Any = None,
        diis_start_cycle: int | None = None,
        level_shift_factor: float | None = None,
        damp_factor: float | None = None,
        fock_last: Any | None = None,
    ) -> Any:
        if h1e is None:
            h1e = self.mf.get_hcore(self.mf.mol)
        if vhf is None:
            vhf = self.mf.get_veff(self.mf.mol, dm)
        if s1e is None:
            s1e = self.mf.get_ovlp(self.mf.mol)
        if dm is None:
            dm = self.mf.make_rdm1()

        base_fock = _stack_spin_matrices(h1e + vhf[0], h1e + vhf[1])
        if cycle >= 0:
            self._solve_multipliers(base_fock, s1e)

        operator = self._constraint_operator(self.v_lagrange)
        operator_like = _to_backend_array(h1e, operator)
        h1e_eff = h1e + operator_like
        return self._orig_get_fock(
            h1e=h1e_eff,
            s1e=s1e,
            vhf=vhf,
            dm=dm,
            cycle=cycle,
            diis=diis,
            diis_start_cycle=diis_start_cycle,
            level_shift_factor=level_shift_factor,
            damp_factor=damp_factor,
            fock_last=fock_last,
        )

    def _refine_constraints_after_scf(self) -> tuple[np.ndarray, float]:
        """Refine multipliers/density after SCF convergence until cDFT converges."""
        max_steps = max(int(self.solver_options.max_cycle), 1)
        h1e = self.mf.get_hcore(self.mf.mol)
        s1e = self.mf.get_ovlp(self.mf.mol)
        dm_backend = self.mf.make_rdm1()
        energy = float(getattr(self.mf, "e_tot", 0.0))

        for step in range(max_steps):
            vhf = self.mf.get_veff(self.mf.mol, dm_backend)
            base_fock = _stack_spin_matrices(h1e + vhf[0], h1e + vhf[1])
            self._solve_multipliers(base_fock, s1e)

            operator = self._constraint_operator(self.v_lagrange)
            operator_like = _to_backend_array(base_fock[0], operator)
            constrained_fock = _stack_spin_matrices(
                base_fock[0] + operator_like,
                base_fock[1] + operator_like,
            )
            mo_energy, mo_coeff = self.mf.eig(constrained_fock, s1e)
            mo_occ = self.mf.get_occ(mo_energy, mo_coeff)
            dm_backend = self.mf.make_rdm1(mo_coeff, mo_occ)

            # Keep wrapped MF state in sync with refined constrained solution.
            self.mf.mo_energy = mo_energy
            self.mf.mo_coeff = mo_coeff
            self.mf.mo_occ = mo_occ

            dm_np = _to_numpy_array(dm_backend).astype(float, copy=False)
            residuals = evaluate_constraint_residuals(dm_np[0] + dm_np[1], self.constraint_system)
            residual_norm = float(np.linalg.norm(residuals))
            self._last_residuals = np.asarray(residuals, dtype=float)
            self.solver_state["residual_norm"] = residual_norm
            self.solver_state["refinement_steps"] = step + 1

            if residual_norm <= self.solver_options.conv_tol_constraint:
                break

        vhf_final = self.mf.get_veff(self.mf.mol, dm_backend)
        if hasattr(self.mf, "energy_tot"):
            energy = float(self.mf.energy_tot(dm=dm_backend, h1e=h1e, vhf=vhf_final))
            self.mf.e_tot = energy
        dm_final = _to_numpy_array(dm_backend).astype(float, copy=False)
        return dm_final, energy

    def constraint_values(self, dm: np.ndarray | None = None) -> dict[str, float]:
        """Return user-facing constraint values keyed by constraint name."""
        if dm is None:
            dm = _to_numpy_array(self.mf.make_rdm1()).astype(float, copy=False)
        raw = evaluate_constraint_values(dm[0] + dm[1], self.constraint_system)
        shown = report_constraint_values(raw, self.constraint_system)
        return {
            name: float(value)
            for name, value in zip(self.constraint_system.names, shown, strict=True)
        }

    def constraint_residuals(self, dm: np.ndarray | None = None) -> dict[str, float]:
        """Return user-facing residual values keyed by constraint name."""
        if dm is None:
            dm = _to_numpy_array(self.mf.make_rdm1()).astype(float, copy=False)
        raw = evaluate_constraint_residuals(dm[0] + dm[1], self.constraint_system)
        shown = report_constraint_residuals(raw, self.constraint_system)
        return {
            name: float(value)
            for name, value in zip(self.constraint_system.names, shown, strict=True)
        }

    def get_canonical_mo(self) -> tuple[np.ndarray, np.ndarray]:
        """Return canonical orbitals from unconstrained Fock at constrained density."""
        dm = self.mf.make_rdm1()
        h1e = self.mf.get_hcore(self.mf.mol)
        vhf = self.mf.get_veff(self.mf.mol, dm)
        overlap = self.mf.get_ovlp(self.mf.mol)
        fock = np.stack([h1e + vhf[0], h1e + vhf[1]])
        return self.mf.eig(fock, overlap)

    def newton(self) -> "CDFT":
        """Switch to coupled Newton-KKT mode and return self."""
        self.solver_options.mode = "newton_kkt"
        return self

    def kernel(self, *args: Any, **kwargs: Any) -> float:
        """Run constrained SCF and enforce both SCF and constraint convergence."""
        energy = float(self.mf.kernel(*args, **kwargs))
        dm = _to_numpy_array(self.mf.make_rdm1()).astype(float, copy=False)
        residuals = evaluate_constraint_residuals(dm[0] + dm[1], self.constraint_system)
        residual_norm = float(np.linalg.norm(residuals))
        scf_converged = bool(self.mf.converged)
        cdft_converged = bool(residual_norm <= self.solver_options.conv_tol_constraint)

        # If SCF converged first (common with converged initial guesses), keep
        # iterating multipliers/density until constraint tolerance is satisfied.
        if scf_converged and not cdft_converged:
            dm, energy = self._refine_constraints_after_scf()
            residuals = evaluate_constraint_residuals(dm[0] + dm[1], self.constraint_system)
            residual_norm = float(np.linalg.norm(residuals))
            cdft_converged = bool(residual_norm <= self.solver_options.conv_tol_constraint)

        self._last_residuals = np.asarray(residuals, dtype=float)
        self.solver_state["residual_norm"] = residual_norm
        self.converged = bool(scf_converged and cdft_converged)
        if not self.converged:
            diagnostics = [
                f"scf_converged={scf_converged}",
                f"constraint_residual_norm={residual_norm:.3e}",
                f"constraint_tol={self.solver_options.conv_tol_constraint:.3e}",
            ]
            raise ConvergenceError(
                "cDFT kernel did not satisfy convergence criteria.",
                diagnostics=diagnostics,
            )
        return energy
