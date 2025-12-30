from datetime import datetime

import pytest
from pytest_mock import MockerFixture

from tests.helpers import ColumnCreator
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.resource.resource import Column, NullResource, Resource
from vivarium.framework.values import AttributePipeline, Pipeline, ValueModifier, ValueSource


class TestResourceGroup:

    component = ColumnCreator()

    @pytest.mark.parametrize("resource_type", ["column", "resource"])
    def test_resource_group(self, resource_type: str, mocker: MockerFixture) -> None:
        resources: list[Column] | Resource
        if resource_type == "column":
            resources = [Column(f"resource_{i}", self.component) for i in range(5)]
        else:
            resources = Resource("test", "some_resource", self.component)
        r_dependencies = [
            AttributePipeline("an_interesting_attribute", None),
            Pipeline("baz"),
            RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap(), mocker.Mock()),
            ValueSource(Pipeline("foo"), lambda: 1, None),
        ]

        rg = ResourceGroup(resources, r_dependencies)

        assert rg.component == self.component
        assert rg.type == "column" if resource_type == "column" else "test"
        assert (
            rg.names == [f"column.{res.name}" for res in resources]
            if isinstance(resources, list)
            else ["test.some_resource"]
        )
        assert rg.initializer == self.component.on_initialize_simulants
        assert rg.dependencies == [
            "attribute.an_interesting_attribute",
            "value.baz",
            "stream.bar",
            "value_source.foo",
        ]
        assert list(rg) == rg.names


@pytest.mark.parametrize(
    "resource, has_initializer",
    [
        (Pipeline("foo", ColumnCreator()), False),
        (AttributePipeline("foo", ColumnCreator()), False),
        (ValueSource(Pipeline("bar", ColumnCreator()), lambda: 1, ColumnCreator()), False),
        (ValueModifier(Pipeline("baz", ColumnCreator()), lambda: 1, ColumnCreator()), False),
        (Column("foo", ColumnCreator()), True),
        (
            RandomnessStream("bar", lambda: datetime.now(), 1, IndexMap(), ColumnCreator()),
            False,
        ),
        (NullResource(0, ColumnCreator()), True),
    ],
)
def test_resource_group_is_initializer(resource: Resource, has_initializer: bool) -> None:
    rg = ResourceGroup(resource, [Resource("test", "bar", None)])
    assert rg.is_initialized == has_initializer


def test_resource_group_with_no_resources() -> None:
    with pytest.raises(ResourceError, match="must have at least one resource"):
        _ = ResourceGroup([], [Resource("test", "foo", None)])


def test_resource_group_with_multiple_components() -> None:
    # This test is not terribly relevant since ResourceGroup now only accepts a list
    # of Columns or a single Resource. We keep it around in case that changes.
    resources = [
        Column("foo", ColumnCreator()),
        Column("bar", ColumnCreator()),
    ]

    with pytest.raises(ResourceError, match="resources must have the same component"):
        _ = ResourceGroup(resources, [])


def test_resource_group_with_multiple_resource_types() -> None:

    # This test is not terribly relevant since ResourceGroup now only accepts a list
    # of Columns or a single Resource. We keep it around in case that changes.
    component = ColumnCreator()
    resources = [
        ValueModifier(Pipeline("foo"), lambda: 1, component),
        ValueSource(Pipeline("bar"), lambda: 2, component),
    ]

    with pytest.raises(ResourceError, match="resources must be of the same type"):
        _ = ResourceGroup(resources, [])  # type: ignore[arg-type]


from pytest_mock import MockerFixture


def test_set_dependencies(mocker: MockerFixture) -> None:
    some_attribute = AttributePipeline("some_attribute", mocker.Mock())
    some_other_attribute = AttributePipeline("some_other_attribute", mocker.Mock())
    resource = Resource("test", "some_resource", mocker.Mock())
    dependencies: list[AttributePipeline | str] = [
        some_attribute,
        "some_other_attribute",
    ]

    rg = ResourceGroup(resource, dependencies)
    assert rg._dependencies == [some_attribute, "some_other_attribute"]
    # Mock the attribute pipelines dict
    attribute_pipelines = {
        "some_attribute": some_attribute,
        "some_other_attribute": some_other_attribute,
    }
    rg.set_dependencies(attribute_pipelines)
    # Check that the 'some_other_attribute' string has been replaced by the pipeline
    assert rg._dependencies == [some_attribute, some_other_attribute]
