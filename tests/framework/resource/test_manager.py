from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime

import pytest
import pytest_mock

from vivarium import Component
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import Resource, ResourceManager
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.manager import NULL_RESOURCE_TYPE, RESOURCE_TYPES
from vivarium.framework.values import Pipeline


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


@pytest.mark.parametrize("r_type", RESOURCE_TYPES, ids=lambda x: f"r_type_{x}")
def test_resource_manager_get_resource_group(r_type: str, manager: ResourceManager) -> None:
    component = ResourceProducer("base")
    r_names = ["foo"]
    r_producer = component.producer
    r_dependencies: list[str | Resource] = []

    group = manager._get_resource_group(r_type, r_names, r_producer, r_dependencies)

    assert group.type == r_type
    assert group.names == [f"{r_type}.foo"]
    assert group.producer == component.producer
    assert not group.dependencies


def test_resource_manager_get_resource_group_null(manager: ResourceManager) -> None:
    component = ResourceProducer("base")
    r_names: list[str] = []
    r_producer = component.producer
    r_dependencies: list[str | Resource] = []

    group_1 = manager._get_resource_group("column", r_names, r_producer, r_dependencies)
    group_2 = manager._get_resource_group("column", r_names, r_producer, r_dependencies)

    assert group_1.type == NULL_RESOURCE_TYPE
    assert group_1.names == [f"{NULL_RESOURCE_TYPE}.0"]
    assert group_1.producer == component.producer
    assert not group_1.dependencies

    assert group_2.type == NULL_RESOURCE_TYPE
    assert group_2.names == [f"{NULL_RESOURCE_TYPE}.1"]
    assert group_2.producer == component.producer
    assert not group_2.dependencies


def test_resource_manager_add_resources_bad_type(manager: ResourceManager) -> None:
    c = ResourceProducer("base")
    r_type = "unknown"
    r_names = [str(i) for i in range(5)]
    r_producer = c.producer
    r_dependencies: list[str | Resource] = []

    with pytest.raises(ResourceError, match="Unknown resource type"):
        manager.add_resources(r_type, r_names, r_producer, r_dependencies)


def test_resource_manager_add_resources_multiple_producers(manager: ResourceManager) -> None:
    c1 = ResourceProducer("1")
    c2 = ResourceProducer("2")
    r_type = "column"
    r1_names = [str(i) for i in range(5)]
    r2_names = [str(i) for i in range(5, 10)] + ["1"]
    r1_producer = c1.producer
    r2_producer = c2.producer
    r_dependencies: list[str | Resource] = []

    manager.add_resources(r_type, r1_names, r1_producer, r_dependencies)
    with pytest.raises(ResourceError, match="producers for column.1"):
        manager.add_resources(r_type, r2_names, r2_producer, r_dependencies)


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    c = ResourceProducer("test")

    manager.add_resources("column", ["c_1"], c.producer, [randomness_stream])
    manager.add_resources("stream", [randomness_stream.key], c.producer, ["c_1"])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    c = ResourceProducer("test")
    pipeline = Pipeline("some_pipeline")

    manager.add_resources("column", ["c_1"], c.producer, [randomness_stream])
    manager.add_resources("value", [pipeline.name], c.producer, ["c_1"])
    manager.add_resources("stream", [randomness_stream.key], c.producer, [pipeline])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    c = ResourceProducer("test")

    for i in range(10):
        manager.add_resources("column", [f"c_{i}"], c.producer, [f"c_{i%10}"])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager) -> None:
    for i in range(9, 0, -1):
        manager.add_resources(
            "column", [f"c_{i}"], ResourceProducer(f"p_{i}").producer, [f"c_{i - 1}"]
        )
    manager.add_resources("column", ["c_0"], ResourceProducer("producer_0").producer, [])

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

    assert n.index("(stream.B)") < n.index(f"({NULL_RESOURCE_TYPE}.0)")


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

    manager.add_resources("column", ["D"], producers[3], [stream, pipeline])
    manager.add_resources("stream", ["B"], producers[1], ["A"])
    manager.add_resources("value", ["C"], producers[2], ["A"])
    manager.add_resources("column", ["A"], producers[0], [])
    manager.add_resources("column", [], producers[4], [stream])

    return producers
