from datetime import datetime

import pytest

from tests.helpers import ColumnCreator, ColumnRequirer
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column, NullResource, Resource
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


def test_resource_group() -> None:
    component = ColumnCreator()
    resources = [Column(str(i), component) for i in range(5)]
    r_dependencies = [
        Column("an_interesting_column", None),
        Pipeline("baz"),
        RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()),
        ValueSource(Pipeline("foo"), lambda: 1, None),
    ]

    rg = ResourceGroup(resources, r_dependencies)

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


@pytest.mark.parametrize(
    "resource, has_initializer",
    [
        (Pipeline("foo"), False),
        (ValueSource(Pipeline("bar"), lambda: 1, ColumnCreator()), False),
        (ValueModifier(Pipeline("baz"), lambda: 1, ColumnCreator()), False),
        (Column("foo", ColumnCreator()), True),
        (RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()), False),
        (NullResource(0, ColumnCreator()), True),
    ],
)
def test_resource_group_is_initializer(resource: Resource, has_initializer: bool) -> None:
    rg = ResourceGroup([resource], [Column("bar", None)])
    assert rg.is_initialized == has_initializer


def test_resource_group_with_no_resources() -> None:
    with pytest.raises(ResourceError, match="must have at least one resource"):
        _ = ResourceGroup([], [Column("foo", None)])


def test_resource_group_with_multiple_components() -> None:
    resources = [
        ValueModifier(Pipeline("foo"), lambda: 1, ColumnCreator()),
        ValueSource(Pipeline("bar"), lambda: 2, ColumnRequirer()),
    ]

    with pytest.raises(ResourceError, match="resources must have the same component"):
        _ = ResourceGroup(resources, [])


def test_resource_group_with_multiple_resource_types() -> None:
    component = ColumnCreator()
    resources = [
        ValueModifier(Pipeline("foo"), lambda: 1, component),
        ValueSource(Pipeline("bar"), lambda: 2, component),
    ]

    with pytest.raises(ResourceError, match="resources must be of the same type"):
        _ = ResourceGroup(resources, [])
