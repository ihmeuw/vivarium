from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

import pytest
import pytest_mock

from vivarium import Component
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import ResourceManager
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, NullResource
from vivarium.framework.values import MissingValueSource, Pipeline, ValueModifier, ValueSource


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

    def producer(self, _simulant_data: SimulantData) -> None:
        pass


@pytest.mark.parametrize(
    "resource_class, type_string",
    [
        (Pipeline, "value"),
        (ValueSource, "value_source"),
        (MissingValueSource, "missing_value_source"),
        (ValueModifier, "value_modifier"),
        (Column, "column"),
        (NullResource, "null"),
    ],
    ids=lambda x: {x.__name__ if isinstance(x, type) else x},
)
def test_resource_manager_get_resource_group(
    resource_class: type, type_string: str, manager: ResourceManager
) -> None:
    producer = ResourceProducer("base").producer

    group = manager._get_resource_group([resource_class("foo")], producer, [])

    assert group.type == type_string
    assert group.names == [f"{type_string}.foo"]
    assert group.producer == producer
    assert not group.dependencies


def test_resource_manager_get_resource_group_null(manager: ResourceManager) -> None:
    producer = ResourceProducer("base").producer

    group_1 = manager._get_resource_group([], producer, [])
    group_2 = manager._get_resource_group([], producer, [])

    assert group_1.type == "null"
    assert group_1.names == ["null.0"]
    assert group_1.producer == producer
    assert not group_1.dependencies

    assert group_2.type == "null"
    assert group_2.names == ["null.1"]
    assert group_2.producer == producer
    assert not group_2.dependencies


def test_resource_manager_add_resources_multiple_producers(manager: ResourceManager) -> None:
    r1 = [str(i) for i in range(5)]
    r2 = [str(i) for i in range(5, 10)] + ["1"]

    manager.add_resources(r1, ResourceProducer("1").producer, [])
    with pytest.raises(ResourceError, match="producers for column.1"):
        manager.add_resources(r2, ResourceProducer("2").producer, [])


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    manager.add_resources(["c_1"], ResourceProducer("1").producer, [randomness_stream])
    manager.add_resources([randomness_stream], ResourceProducer("2").producer, ["c_1"])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    pipeline = Pipeline("some_pipeline")

    manager.add_resources(["c_1"], ResourceProducer("1").producer, [randomness_stream])
    manager.add_resources([pipeline], ResourceProducer("2").producer, ["c_1"])
    manager.add_resources([randomness_stream], ResourceProducer("3").producer, [pipeline])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    for i in range(10):
        manager.add_resources([f"c_{i}"], ResourceProducer("1").producer, [f"c_{i % 10}"])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager) -> None:
    for i in range(9, 0, -1):
        manager.add_resources([f"c_{i}"], ResourceProducer(f"p_{i}").producer, [f"c_{i - 1}"])
    manager.add_resources(["c_0"], ResourceProducer("producer_0").producer, [])

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
    producers = {i: ResourceProducer(f"test_{i}").producer for i in range(5)}

    stream = RandomnessStream("B", lambda: datetime.now(), 1, IndexMap())
    pipeline = Pipeline("C")

    manager.add_resources(["D"], producers[3], [stream, pipeline])
    manager.add_resources([stream], producers[1], ["A"])
    manager.add_resources([pipeline], producers[2], ["A"])
    manager.add_resources(["A"], producers[0], [])
    manager.add_resources([], producers[4], [stream])

    return producers
