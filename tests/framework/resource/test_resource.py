from datetime import datetime

import pytest

from tests.helpers import ColumnCreator
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import Resource
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column, NullResource
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


def test_resource_id() -> None:
    resource = Resource("value_source", "test")
    assert resource.resource_id == "value_source.test"


@pytest.mark.parametrize(
    "resource, is_initialized",
    [
        (Pipeline("foo"), False),
        (ValueSource(Pipeline("bar"), lambda: 1), False),
        (ValueModifier(Pipeline("baz"), lambda: 1), False),
        (Column("foo"), True),
        (RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap()), False),
        (NullResource(0), True),
    ],
)
def test_resource_is_initialized(resource: Resource, is_initialized: bool) -> None:
    assert resource.is_initialized == is_initialized


def test_resource_component() -> None:
    component = ColumnCreator()
    resource = Column("foo")

    _ = ResourceGroup(component, [resource], [])
    assert resource.component == component
