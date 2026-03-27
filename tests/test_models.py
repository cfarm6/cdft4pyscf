"""Model validation tests."""

from __future__ import annotations

import pytest

from cdft4pyscf.models import Constraint, FragmentTerm, ProjectorSpec, RunRequest


def test_fragment_term_validates_atom_rules() -> None:
    """Fragment terms enforce unique non-negative atom indices."""
    with pytest.raises(ValueError, match="unique"):
        FragmentTerm(atoms=[0, 0], coeff=1.0)
    with pytest.raises(ValueError, match="non-negative"):
        FragmentTerm(atoms=[-1], coeff=1.0)
    with pytest.raises(ValueError, match="cannot be zero"):
        FragmentTerm(atoms=[0], coeff=0.0)


def test_constraint_supports_difference_fragments() -> None:
    """Difference constraints use signed fragment coefficients."""
    constraint = Constraint(
        name="delta",
        fragments=[
            FragmentTerm(atoms=[0], coeff=1.0),
            FragmentTerm(atoms=[1], coeff=-1.0),
        ],
        target=0.0,
        target_type="electrons",
    )
    expected_terms = 2
    assert len(constraint.fragments) == expected_terms


def test_constraint_accepts_projector_spec() -> None:
    """Constraint should carry projector backend metadata."""
    constraint = Constraint(
        name="n_a",
        fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
        target=1.0,
        target_type="electrons",
        projector=ProjectorSpec(method="minao"),
    )
    assert constraint.projector.method == "minao"


def test_run_request_enforces_unique_constraint_names() -> None:
    """RunRequest rejects duplicate constraint names."""
    with pytest.raises(ValueError, match="unique"):
        RunRequest(
            atom="H 0 0 0; H 0 0 0.74",
            constraints=[
                Constraint(
                    name="x",
                    fragments=[FragmentTerm(atoms=[0], coeff=1.0)],
                    target=1.0,
                ),
                Constraint(
                    name="x",
                    fragments=[FragmentTerm(atoms=[1], coeff=1.0)],
                    target=1.0,
                ),
            ],
        )
