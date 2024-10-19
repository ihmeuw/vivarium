from __future__ import annotations

from datetime import datetime

import pytest

from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


def dummy_producer() -> str:
    return "resources!"


def test_resource_group() -> None:
    resources = [ValueModifier(str(i)) for i in range(5)]
    r_dependencies = [
        Column("an_interesting_column"),
        Pipeline("baz"),
        RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()),
        ValueSource("foo"),
    ]

    rg = ResourceGroup(resources, dummy_producer, r_dependencies)

    assert rg.type == "value_modifier"
    assert rg.names == [f"value_modifier.{i}" for i in range(5)]
    assert rg.producer == dummy_producer
    assert rg.dependencies == [
        "column.an_interesting_column",
        "value.baz",
        "stream.bar",
        "value_source.foo",
    ]
    assert list(rg) == rg.names


def test_resource_group_with_no_resources() -> None:
    with pytest.raises(ResourceError, match="must have at least one resource"):
        _ = ResourceGroup([], dummy_producer, [Column("foo")])


def test_resource_group_with_multiple_resource_types() -> None:
    resources = [ValueModifier("foo"), ValueSource("bar")]

    with pytest.raises(ResourceError, match="resources must be of the same type"):
        _ = ResourceGroup(resources, dummy_producer)
