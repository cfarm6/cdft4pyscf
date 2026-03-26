"""Model validation tests."""

from __future__ import annotations

import pytest

from cdft4pyscf.models import ConstraintSpec, RegionSpec


def test_constraint_region_rules() -> None:
    """Constraint kinds enforce region usage policy."""
    with pytest.raises(ValueError, match="require a region"):
        ConstraintSpec(name="a", kind="electron_number", target=1.0)

    with pytest.raises(ValueError, match="require a region"):
        ConstraintSpec(name="b", kind="net_charge", target=0.0)

    with pytest.raises(ValueError, match="single region"):
        ConstraintSpec(
            name="b",
            kind="net_charge",
            target=0.0,
            region=[RegionSpec(name="frag", atom_indices=[0])],
        )


def test_constraint_accepts_single_region_object() -> None:
    """Electron-number constraints accept an inline region."""
    spec = ConstraintSpec(
        name="n_frag",
        kind="electron_number",
        target=1.0,
        region=RegionSpec(name="frag", atom_indices=[0]),
    )
    assert isinstance(spec.region, RegionSpec)


def test_constraint_accepts_region_list() -> None:
    """Electron-number constraints can combine multiple regions."""
    spec = ConstraintSpec(
        name="n_frags",
        kind="electron_number",
        target=2.0,
        region=[
            RegionSpec(name="frag_a", atom_indices=[0]),
            RegionSpec(name="frag_b", atom_indices=[1]),
        ],
    )
    assert isinstance(spec.region, list)


def test_net_charge_accepts_single_region_object() -> None:
    """Net-charge constraints accept a single inline region."""
    spec = ConstraintSpec(
        name="q_frag",
        kind="net_charge",
        target=0.0,
        region=RegionSpec(name="frag", atom_indices=[0]),
    )
    assert isinstance(spec.region, RegionSpec)
