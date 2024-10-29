from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Any

import pytest
import pytest_mock

from vivarium import Component
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import ResourceManager
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, NullResource
from vivarium.framework.values import Pipeline, ValueModifier, ValueSource


@pytest.fixture
def manager(mocker: pytest_mock.MockFixture) -> ResourceManager:
    manager = ResourceManager()
    manager.logger = mocker.Mock()
    return manager


@pytest.fixture
def randomness_stream() -> RandomnessStream:
    return RandomnessStream("stream.1", lambda: datetime.now(), 1, IndexMap())


class ResourceProducer(Component):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def initializer(self, _simulant_data: SimulantData) -> None:
        pass


@pytest.mark.parametrize(
    "resource_class, init_args, type_string, is_initializer",
    [
        (Pipeline, ["foo"], "value", False),
        (ValueSource, [Pipeline("foo"), lambda: 1], "value_source", False),
        (ValueModifier, [Pipeline("foo"), lambda: 1], "value_modifier", False),
        (Column, ["foo"], "column", True),
        (NullResource, ["foo"], "null", True),
    ],
    ids=lambda x: [x.__name__ if isinstance(x, type) else x],
)
def test_resource_manager_get_resource_group(
    resource_class: type,
    init_args: list[Any],
    type_string: str,
    is_initializer: bool,
    manager: ResourceManager,
) -> None:
    initializer = ResourceProducer("base").initializer

    group = manager._get_resource_group(
        [resource_class(*init_args)], [], initializer if is_initializer else None
    )

    assert group.type == type_string
    assert group.names == [r.resource_id for r in group._resources.values()]
    assert not group.dependencies
    assert group.is_initializer == is_initializer
    if is_initializer:
        assert group.initializer == initializer
    else:
        with pytest.raises(ResourceError, match="does not have an initializer"):
            _ = group.initializer


def test_resource_manager_get_resource_group_null(manager: ResourceManager) -> None:
    initializer = ResourceProducer("base").initializer

    group_1 = manager._get_resource_group([], [], initializer)
    group_2 = manager._get_resource_group([], [], initializer)

    assert group_1.type == "null"
    assert group_1.names == ["null.0"]
    assert group_1.initializer == initializer
    assert not group_1.dependencies

    assert group_2.type == "null"
    assert group_2.names == ["null.1"]
    assert group_2.initializer == initializer
    assert not group_2.dependencies


def test_resource_manager_add_same_column_twice(manager: ResourceManager) -> None:
    r1 = [str(i) for i in range(5)]
    r2 = [str(i) for i in range(5, 10)] + ["1"]

    manager.add_resources(r1, [], ResourceProducer("1").initializer)
    with pytest.raises(ResourceError, match="initializers for column.1"):
        manager.add_resources(r2, [], ResourceProducer("2").initializer)


def test_resource_manager_add_same_pipeline_twice(manager: ResourceManager) -> None:
    r1 = [Pipeline(str(i)) for i in range(5)]
    r2 = [Pipeline(str(i)) for i in range(5, 10)] + [Pipeline("1")]

    manager.add_resources(r1, [], None)
    with pytest.raises(ResourceError, match="registered more than once"):
        manager.add_resources(r2, [], None)


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    manager.add_resources(["c_1"], [randomness_stream], ResourceProducer("1").initializer)
    manager.add_resources([randomness_stream], ["c_1"], None)

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    pipeline = Pipeline("some_pipeline")

    manager.add_resources(["c_1"], [randomness_stream], ResourceProducer("1").initializer)
    manager.add_resources([pipeline], ["c_1"], None)
    manager.add_resources([randomness_stream], [pipeline], None)

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    for i in range(10):
        manager.add_resources([f"c_{i}"], [f"c_{i % 10}"], ResourceProducer("1").initializer)

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager) -> None:
    for i in range(9, 0, -1):
        manager.add_resources(
            [f"c_{i}"], [f"c_{i - 1}"], ResourceProducer(f"p_{i}").initializer
        )
    manager.add_resources(["c_0"], [], ResourceProducer("producer_0").initializer)

    for i, resource in enumerate(manager.sorted_nodes):
        assert str(resource) == f"(column.c_{i})"


def test_resource_manager_sorted_nodes_acyclic(manager: ResourceManager) -> None:
    _add_resources(manager)

    n = [str(node) for node in manager.sorted_nodes]

    assert n.index("(column.A)") < n.index("(stream.B)")
    assert n.index("(column.A)") < n.index("(value.C)")
    assert n.index("(column.A)") < n.index("(column.D)")

    assert n.index("(stream.B)") < n.index("(column.D)")
    assert n.index("(value.C)") < n.index("(column.D)")

    assert n.index("(stream.B)") < n.index(f"(null.0)")


def test_get_population_initializers(manager: ResourceManager) -> None:
    producers = _add_resources(manager)
    initializers = manager.get_population_initializers()

    assert len(initializers) == 3
    assert initializers[0] == producers[0]
    assert producers[3] in initializers
    assert producers[4] in initializers


####################
# Helper functions #
####################


def _add_resources(manager: ResourceManager) -> Mapping[int, Callable[[SimulantData], None]]:
    producers = {i: ResourceProducer(f"test_{i}").initializer for i in range(5)}

    stream = RandomnessStream("B", lambda: datetime.now(), 1, IndexMap())
    pipeline = Pipeline("C")

    manager.add_resources(["D"], [stream, pipeline], producers[3])
    manager.add_resources([stream], ["A"], None)
    manager.add_resources([pipeline], ["A"], None)
    manager.add_resources(["A"], [], producers[0])
    manager.add_resources([], [stream], producers[4])

    return producers
