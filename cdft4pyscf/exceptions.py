"""Custom exceptions for the cdft4pyscf package."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class CdftError(Exception):
    """Base package exception."""

    message: str

    def __str__(self) -> str:
        """Return the human readable message."""
        return self.message


@dataclass(slots=True)
class ConvergenceError(CdftError):
    """Raised when the solver fails to satisfy convergence criteria."""

    diagnostics: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        """Format convergence diagnostics for error display."""
        if not self.diagnostics:
            return self.message
        diagnostics = "; ".join(self.diagnostics)
        return f"{self.message} ({diagnostics})"


@dataclass(slots=True)
class BackendUnavailableError(CdftError):
    """Raised when a requested backend cannot be initialized."""
