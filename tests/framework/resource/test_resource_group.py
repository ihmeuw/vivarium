from __future__ import annotations

from datetime import datetime

import pytest

from tests.helpers import ColumnCreator
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column, NullResource, Resource
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


def test_resource_group() -> None:
    component = ColumnCreator()
    resources = [Column(str(i)) for i in range(5)]
    r_dependencies = [
        Column("an_interesting_column"),
        Pipeline("baz"),
        RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()),
        ValueSource(Pipeline("foo"), lambda: 1),
    ]

    rg = ResourceGroup(component, resources, r_dependencies)

    assert rg.component == component
    assert rg.type == "column"
    assert rg.names == [f"column.{i}" for i in range(5)]
    assert rg.initializer == component.on_initialize_simulants
    assert rg.dependencies == [
        "column.an_interesting_column",
        "value.baz",
        "stream.bar",
        "value_source.foo",
    ]
    assert list(rg) == rg.names
    assert all(r.resource_group == rg for r in resources)


@pytest.mark.parametrize(
    "resource, has_initializer",
    [
        (Pipeline("foo"), False),
        (ValueSource(Pipeline("bar"), lambda: 1), False),
        (ValueModifier(Pipeline("baz"), lambda: 1), False),
        (Column("foo"), True),
        (RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()), False),
        (NullResource(0), True),
    ],
)
def test_resource_group_is_initializer(resource: Resource, has_initializer: bool) -> None:
    rg = ResourceGroup(ColumnCreator(), [resource], [Column("bar")])
    assert rg.is_initialized == has_initializer


def test_resource_group_no_initializer_raises_when_called() -> None:
    resources = [ValueModifier(Pipeline("foo"), lambda: 1)]
    rg = ResourceGroup(ColumnCreator(), resources, [Column("bar")])

    with pytest.raises(ResourceError, match="is not an initializer"):
        _ = rg.initializer


def test_resource_group_with_no_resources() -> None:
    with pytest.raises(ResourceError, match="must have at least one resource"):
        _ = ResourceGroup(ColumnCreator(), [], [Column("foo")])


def test_resource_group_with_multiple_resource_types() -> None:
    resources = [
        ValueModifier(Pipeline("foo"), lambda: 1),
        ValueSource(Pipeline("bar"), lambda: 2),
    ]

    with pytest.raises(ResourceError, match="resources must be of the same type"):
        _ = ResourceGroup(ColumnCreator(), resources, [])
