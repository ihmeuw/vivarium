from __future__ import annotations

from datetime import datetime

import pytest

from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


def dummy_initializer(_simulant_data: SimulantData) -> None:
    pass


def test_resource_group() -> None:
    resources = [Column(str(i)) for i in range(5)]
    r_dependencies = [
        Column("an_interesting_column"),
        Pipeline("baz"),
        RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()),
        ValueSource("foo"),
    ]

    rg = ResourceGroup(resources, r_dependencies, dummy_initializer)

    assert rg.type == "column"
    assert rg.names == [f"column.{i}" for i in range(5)]
    assert rg.initializer == dummy_initializer
    assert rg.dependencies == [
        "column.an_interesting_column",
        "value.baz",
        "stream.bar",
        "value_source.foo",
    ]
    assert list(rg) == rg.names


def test_resource_group_is_initializer() -> None:
    resources = [ValueModifier("foo")]
    rg = ResourceGroup(resources, [Column("bar")])

    with pytest.raises(ResourceError, match="does not have an initializer"):
        _ = rg.initializer


def test_resource_group_with_no_resources() -> None:
    with pytest.raises(ResourceError, match="must have at least one resource"):
        _ = ResourceGroup([], [Column("foo")])


def test_resource_group_with_multiple_resource_types() -> None:
    resources = [ValueModifier("foo"), ValueSource("bar")]

    with pytest.raises(ResourceError, match="resources must be of the same type"):
        _ = ResourceGroup(resources, [])
