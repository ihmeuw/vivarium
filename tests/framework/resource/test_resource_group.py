from __future__ import annotations

import pytest

from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import Resource
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.group import ResourceGroup
from vivarium.framework.values import Pipeline


def dummy_producer() -> str:
    return "resources!"


def test_resource_group() -> None:
    r_type = "column"
    r_names = [str(i) for i in range(5)]
    r_producer = dummy_producer
    r_dependencies: list[str | Pipeline | RandomnessStream | Resource] = [
        "an_interesting_column",
        Pipeline("baz"),
        RandomnessStream("bar", lambda x: x, 1, IndexMap()),
        Resource("value_source", "foo"),
    ]

    rg = ResourceGroup(r_type, r_names, r_producer, r_dependencies)

    assert rg.type == r_type
    assert rg.names == [f"{r_type}.{name}" for name in r_names]
    assert rg.producer == dummy_producer
    assert rg.dependencies == [
        "column.an_interesting_column",
        "value.baz",
        "stream.bar",
        "value_source.foo",
    ]
    assert list(rg) == rg.names


@pytest.mark.parametrize(
    "dependency, expected_key",
    [
        ("an_interesting_column", "column.an_interesting_column"),
        (Pipeline("baz"), "value.baz"),
        (RandomnessStream("bar", lambda x: x, 1, IndexMap()), "stream.bar"),
        (Resource("value_source", "foo"), "value_source.foo"),
    ],
)
def test__get_dependency_key(
    dependency: str | Pipeline | RandomnessStream | Resource, expected_key: str
) -> None:
    key = ResourceGroup._get_dependency_key(dependency)
    assert key == expected_key


def test__get_dependency_key_unknown_type() -> None:
    with pytest.raises(ResourceError, match="unknown type"):
        ResourceGroup._get_dependency_key(1)  # type: ignore [arg-type]
