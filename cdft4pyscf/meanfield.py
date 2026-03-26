"""Class-based constrained UKS mean-field objects."""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np
from pyscf.dft import uks as pyscf_uks
from scipy import optimize

from cdft4pyscf.constraints import (
    ConstraintSystem,
    build_constraint_system,
    evaluate_constraint_residuals,
    evaluate_constraint_values,
    report_constraint_residuals,
    report_constraint_values,
)
from cdft4pyscf.exceptions import BackendUnavailableError, ConvergenceError


def _cupy_module() -> Any | None:
    try:
        cp = importlib.import_module("cupy")
    except Exception:
        return None
    return cp


def _is_cupy_array(value: Any) -> bool:
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
    """Convert a NumPy array to the backend type of reference."""
    if _is_cupy_array(reference):
        cp = _cupy_module()
        if cp is not None:
            return cp.asarray(value)
    return np.asarray(value)


def _stack_spin_matrices(alpha: Any, beta: Any) -> Any:
    """Stack alpha/beta spin matrices while preserving backend arrays when possible."""
    if _is_cupy_array(alpha) and _is_cupy_array(beta):
        cp = _cupy_module()
        if cp is not None:
            return cp.stack([alpha, beta])
    return np.stack([_to_numpy_array(alpha), _to_numpy_array(beta)])


class _CDFTMixin:
    """Mixin that injects cDFT multiplier updates into UKS get_fock()."""

    _keys = frozenset(
        {
            "constraint_system",
            "vc",
            "cdft_conv_tol",
            "cdft_vc_tol",
            "cdft_vc_max_cycle",
            "cdft_inner_calls",
            "cdft_last_values",
            "cdft_last_residuals",
            "cdft_residual_norm",
            "cdft_log_inner_solver",
            "cdft_raise_on_unconverged",
            "cdft_messages",
        }
    )

    def _init_cdft(
        self,
        *,
        constraint_system: ConstraintSystem,
        initial_vc: np.ndarray | list[float] | None = None,
        conv_tol: float = 1e-7,
        vc_tol: float = 1e-6,
        vc_max_cycle: int = 50,
        log_inner_solver: bool = False,
        raise_on_unconverged: bool = True,
    ) -> None:
        self.constraint_system = constraint_system
        n_constraints = len(constraint_system.names)
        if initial_vc is None:
            self.vc = np.zeros(n_constraints, dtype=float)
        else:
            vc = np.asarray(initial_vc, dtype=float)
            if vc.shape != (n_constraints,):
                msg = (
                    "initial_vc length must match number of constraints "
                    f"({n_constraints}); got {int(vc.size)}"
                )
                raise ValueError(msg)
            self.vc = vc

        self.cdft_conv_tol = float(conv_tol)
        self.cdft_vc_tol = float(vc_tol)
        self.cdft_vc_max_cycle = int(vc_max_cycle)
        self.cdft_log_inner_solver = bool(log_inner_solver)
        self.cdft_raise_on_unconverged = bool(raise_on_unconverged)

        self.cdft_inner_calls = 0
        self.cdft_last_values = np.zeros(n_constraints, dtype=float)
        self.cdft_last_residuals = np.zeros(n_constraints, dtype=float)
        self.cdft_residual_norm = float("inf")
        self.cdft_messages: list[str] = []

    def _constraint_operator(self, vc: np.ndarray) -> np.ndarray:
        operator = np.zeros_like(self.constraint_system.operators[0])
        for multiplier, weight in zip(vc, self.constraint_system.operators, strict=True):
            operator = operator + (float(multiplier) * weight)
        return operator

    def _density_from_fock(self: Any, fock: np.ndarray, overlap: np.ndarray) -> np.ndarray:
        mo_energy, mo_coeff = self.eig(fock, overlap)
        mo_occ = self.get_occ(mo_energy, mo_coeff)
        return self.make_rdm1(mo_coeff, mo_occ)

    def _update_vc(self, *, base_fock: np.ndarray, overlap: np.ndarray) -> None:
        if len(self.constraint_system.names) == 0:
            return

        dm_cache: np.ndarray | None = None
        values_cache: np.ndarray | None = None

        def objective(vc: np.ndarray) -> np.ndarray:
            nonlocal dm_cache, values_cache
            operator = self._constraint_operator(vc)
            operator_like = _to_backend_array(base_fock[0], operator)
            constrained_fock = _stack_spin_matrices(
                base_fock[0] + operator_like,
                base_fock[1] + operator_like,
            )
            dm = self._density_from_fock(constrained_fock, overlap)
            values = evaluate_constraint_values(
                _to_numpy_array(dm[0] + dm[1]),
                self.constraint_system,
            )
            dm_cache = dm
            values_cache = values
            return values - self.constraint_system.targets

        result = optimize.root(
            objective,
            self.vc,
            method="hybr",
            options={"xtol": self.cdft_vc_tol, "maxfev": self.cdft_vc_max_cycle},
        )
        self.cdft_inner_calls += int(result.nfev)
        self.vc = np.asarray(result.x, dtype=float)

        if values_cache is None:
            operator = self._constraint_operator(self.vc)
            operator_like = _to_backend_array(base_fock[0], operator)
            constrained_fock = _stack_spin_matrices(
                base_fock[0] + operator_like,
                base_fock[1] + operator_like,
            )
            dm_cache = self._density_from_fock(constrained_fock, overlap)
            values_cache = evaluate_constraint_values(
                _to_numpy_array(dm_cache[0] + dm_cache[1]),
                self.constraint_system,
            )

        self.cdft_last_values = np.asarray(values_cache, dtype=float)
        self.cdft_last_residuals = self.cdft_last_values - self.constraint_system.targets
        self.cdft_residual_norm = float(np.linalg.norm(self.cdft_last_residuals))
        if self.cdft_log_inner_solver:
            msg = (
                f"[cDFT][inner] success={bool(result.success)} status={int(result.status)} "
                f"nfev={int(result.nfev)} residual_norm={self.cdft_residual_norm:.3e}"
            )
            self.cdft_messages.append(msg)

    def get_fock(
        self: Any,
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
            h1e = self.get_hcore(self.mol)
        if vhf is None:
            vhf = self.get_veff(self.mol, dm)
        if s1e is None:
            s1e = self.get_ovlp(self.mol)
        if dm is None:
            dm = self.make_rdm1()

        base_fock = _stack_spin_matrices(h1e + vhf[0], h1e + vhf[1])
        if cycle >= 0:
            self._update_vc(base_fock=base_fock, overlap=s1e)

        operator = self._constraint_operator(self.vc)
        h1e_eff = h1e + _to_backend_array(h1e, operator)
        super_obj: Any = super()
        return super_obj.get_fock(
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

    def constraint_values(self: Any, dm: np.ndarray | None = None) -> dict[str, float]:
        if dm is None:
            dm = self.make_rdm1()
        values = evaluate_constraint_values(_to_numpy_array(dm[0] + dm[1]), self.constraint_system)
        values = report_constraint_values(values, self.constraint_system)
        return {
            name: float(value)
            for name, value in zip(self.constraint_system.names, values, strict=True)
        }

    def constraint_residuals(self: Any, dm: np.ndarray | None = None) -> dict[str, float]:
        if dm is None:
            dm = self.make_rdm1()
        residuals = evaluate_constraint_residuals(
            _to_numpy_array(dm[0] + dm[1]),
            self.constraint_system,
        )
        residuals = report_constraint_residuals(residuals, self.constraint_system)
        return {
            name: float(value)
            for name, value in zip(self.constraint_system.names, residuals, strict=True)
        }

    def multiplier_by_constraint(self) -> dict[str, float]:
        return {
            name: float(value)
            for name, value in zip(self.constraint_system.names, self.vc, strict=True)
        }

    def _refresh_cdft_state_from_dm(self, dm: np.ndarray) -> None:
        """Refresh cached cDFT values/residuals from a total density matrix."""
        total_density = _to_numpy_array(dm[0] + dm[1])
        values = evaluate_constraint_values(total_density, self.constraint_system)
        residuals = evaluate_constraint_residuals(total_density, self.constraint_system)
        self.cdft_last_values = np.asarray(values, dtype=float)
        self.cdft_last_residuals = np.asarray(residuals, dtype=float)
        self.cdft_residual_norm = float(np.linalg.norm(self.cdft_last_residuals))

    def _recover_cdft_after_scf(self: Any, dm: np.ndarray) -> tuple[np.ndarray, bool]:
        """Run bounded post-SCF recovery steps to satisfy cDFT residual tolerance."""
        max_recovery_steps = max(int(getattr(self, "max_cycle", 1)), 1)
        h1e = self.get_hcore(self.mol)
        overlap = self.get_ovlp(self.mol)

        current_dm = dm
        for _ in range(max_recovery_steps):
            vhf = self.get_veff(self.mol, current_dm)
            base_fock = _stack_spin_matrices(h1e + vhf[0], h1e + vhf[1])
            self._update_vc(base_fock=base_fock, overlap=overlap)

            operator = self._constraint_operator(self.vc)
            operator_like = _to_backend_array(base_fock[0], operator)
            constrained_fock = _stack_spin_matrices(
                base_fock[0] + operator_like,
                base_fock[1] + operator_like,
            )
            mo_energy, mo_coeff = self.eig(constrained_fock, overlap)
            mo_occ = self.get_occ(mo_energy, mo_coeff)
            current_dm = self.make_rdm1(mo_coeff, mo_occ)
            self.mo_energy = mo_energy
            self.mo_coeff = mo_coeff
            self.mo_occ = mo_occ
            self._refresh_cdft_state_from_dm(current_dm)
            if self.cdft_residual_norm <= self.cdft_conv_tol:
                break

        if hasattr(self, "energy_tot"):
            vhf = self.get_veff(self.mol, current_dm)
            self.e_tot = float(self.energy_tot(dm=current_dm, h1e=h1e, vhf=vhf))
        converged = bool(self.cdft_residual_norm <= self.cdft_conv_tol)
        return current_dm, converged

    def kernel(self: Any, *args: Any, **kwargs: Any) -> float:
        super_obj: Any = super()
        energy = float(super_obj.kernel(*args, **kwargs))
        dm = self.make_rdm1()
        self._refresh_cdft_state_from_dm(dm)
        scf_converged = bool(self.converged)
        cdft_converged = bool(self.cdft_residual_norm <= self.cdft_conv_tol)
        if scf_converged and not cdft_converged:
            _, cdft_converged = self._recover_cdft_after_scf(dm)
            energy = float(getattr(self, "e_tot", energy))

        self.converged = bool(scf_converged and cdft_converged)
        if (not self.converged) and self.cdft_raise_on_unconverged:
            diagnostics = [
                f"scf_converged={scf_converged}",
                f"constraint_residual_norm={self.cdft_residual_norm:.3e}",
            ]
            raise ConvergenceError(
                "cDFT kernel did not satisfy convergence criteria.",
                diagnostics=diagnostics,
            )
        return energy


class CDFT_UKS(_CDFTMixin, pyscf_uks.UKS):  # noqa: N801
    """Class-first constrained UKS object for PySCF CPU workflows."""

    def __init__(
        self,
        mol: Any,
        *,
        constraints: list[Any],
        initial_vc: np.ndarray | list[float] | None = None,
        conv_tol: float = 1e-7,
        vc_tol: float = 1e-6,
        vc_max_cycle: int = 50,
        log_inner_solver: bool = False,
        raise_on_unconverged: bool = True,
    ) -> None:
        super().__init__(mol)
        overlap = self.get_ovlp(mol)
        system = build_constraint_system(
            constraints=constraints,
            overlap=overlap,
            ao_slices=mol.aoslice_by_atom(),
            atom_charges=np.asarray(mol.atom_charges(), dtype=float),
        )
        self._init_cdft(
            constraint_system=system,
            initial_vc=initial_vc,
            conv_tol=conv_tol,
            vc_tol=vc_tol,
            vc_max_cycle=vc_max_cycle,
            log_inner_solver=log_inner_solver,
            raise_on_unconverged=raise_on_unconverged,
        )


def _build_cdft_uks_gpu_class(base_gpu_uks: type[Any]) -> type[Any]:
    class _CDFT_UKS_GPU(_CDFTMixin, base_gpu_uks):  # noqa: N801
        """Class-first constrained UKS object for GPU4PySCF workflows."""

        def __init__(
            self,
            mol: Any,
            *,
            constraints: list[Any],
            initial_vc: np.ndarray | list[float] | None = None,
            conv_tol: float = 1e-7,
            vc_tol: float = 1e-6,
            vc_max_cycle: int = 50,
            log_inner_solver: bool = False,
            raise_on_unconverged: bool = True,
        ) -> None:
            super().__init__(mol)
            overlap = _to_numpy_array(self.get_ovlp(mol))
            system = build_constraint_system(
                constraints=constraints,
                overlap=overlap,
                ao_slices=mol.aoslice_by_atom(),
                atom_charges=np.asarray(mol.atom_charges(), dtype=float),
            )
            self._init_cdft(
                constraint_system=system,
                initial_vc=initial_vc,
                conv_tol=conv_tol,
                vc_tol=vc_tol,
                vc_max_cycle=vc_max_cycle,
                log_inner_solver=log_inner_solver,
                raise_on_unconverged=raise_on_unconverged,
            )

    _CDFT_UKS_GPU.__name__ = "CDFT_UKS_GPU"
    _CDFT_UKS_GPU.__qualname__ = "CDFT_UKS_GPU"
    return _CDFT_UKS_GPU


try:
    _gpu_uks = importlib.import_module("gpu4pyscf.dft.uks")
except Exception:
    _gpu_uks = None

if _gpu_uks is not None:
    CDFT_UKS_GPU = _build_cdft_uks_gpu_class(_gpu_uks.UKS)
else:

    class CDFT_UKS_GPU:  # noqa: N801
        """Placeholder that raises when GPU4PySCF is unavailable."""

        max_cycle: int
        conv_tol: float
        xc: str
        disp: str
        chkfile: Any
        init_guess: str
        verbose: int

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            msg = "GPU backend requested but gpu4pyscf could not be imported."
            raise BackendUnavailableError(msg)

        def kernel(self, *_args: Any, **_kwargs: Any) -> float:
            """Raise a typed error when GPU kernel execution is unavailable."""
            msg = "GPU backend requested but gpu4pyscf could not be imported."
            raise BackendUnavailableError(msg)
