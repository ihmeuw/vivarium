from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
from pytest_mock import MockerFixture

from tests.helpers import ColumnCreator, ColumnCreatorAndRequirer
from vivarium import Component
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.resource import Resource, ResourceManager
from vivarium.framework.resource.exceptions import ResourceError
from vivarium.framework.resource.resource import Column, Initializer
from vivarium.framework.values import AttributePipeline, Pipeline, ValueModifier, ValueSource


@pytest.fixture
def manager(mocker: MockerFixture) -> ResourceManager:
    manager = ResourceManager()
    manager.logger = mocker.Mock()
    manager._get_attribute_pipelines = mocker.Mock(return_value={})
    return manager


@pytest.fixture
def resource_producers() -> dict[int, ResourceProducer]:
    return {i: ResourceProducer(f"test_{i}") for i in range(5)}


@pytest.fixture
def manager_with_resources(
    manager: ResourceManager,
    resource_producers: dict[int, ResourceProducer],
    mocker: MockerFixture,
) -> ResourceManager:
    a_component = resource_producers[0]
    resource_a = Resource("A", a_component)
    manager.add_resource(resource_a)

    stream = RandomnessStream(
        key="B",
        clock=lambda: datetime.now(),
        seed=1,
        index_map=IndexMap(),
        component=resource_producers[1],
    )
    manager.add_resource(stream)

    resource_c = Resource("C", resource_producers[2], [resource_a])
    manager.add_resource(resource_c)

    d_component = resource_producers[3]
    resource_d = Resource("D", d_component, [stream, resource_c])
    manager.add_resource(resource_d)

    manager._get_current_component_or_manager = mocker.Mock(return_value=d_component)
    manager.add_private_columns(
        initializer=d_component.initialize_D,
        columns=["D"],
        required_resources=[stream, resource_c],
    )

    manager._get_current_component_or_manager = mocker.Mock(return_value=a_component)
    manager.add_private_columns(
        initializer=a_component.initialize_A, columns=["A"], required_resources=[]
    )

    manager._get_current_component_or_manager = mocker.Mock(
        return_value=resource_producers[4]
    )
    manager.add_private_columns(
        initializer=resource_producers[4].initialize_nothing,
        columns=[],
        required_resources=[stream],
    )

    return manager


@pytest.fixture
def randomness_stream() -> RandomnessStream:
    return RandomnessStream(
        "stream.1", lambda: datetime.now(), 1, IndexMap(), component=ColumnCreator()
    )


class ResourceProducer(Component):
    @property
    def name(self) -> str:
        return self._name

    def __init__(self, name: str):
        super().__init__()
        self._name = name

    def initialize_A(self, pop_data: SimulantData) -> None:
        pass

    def initialize_D(self, pop_data: SimulantData) -> None:
        pass

    def initialize_nothing(self, pop_data: SimulantData) -> None:
        pass


@pytest.mark.parametrize(
    "resource_type, resource_creator",
    [
        ("column", lambda name, component: Column(name, component)),
        ("value", lambda name, component: Pipeline(name, component)),
        ("attribute", lambda name, component: AttributePipeline(name, component)),
    ],
)
def test_resource_manager_add_same_resource_twice(
    resource_type: str,
    resource_creator: Callable[[str, Component], Any],
    manager: ResourceManager,
) -> None:
    c1 = ColumnCreator()
    c2 = ColumnCreatorAndRequirer()
    resource = resource_creator("test_resource", c1)

    manager.add_resource(resource)

    duplicate_resource = resource_creator("test_resource", c2)
    error_message = (
        f"Component '{c2.name}' is attempting to register resource"
        f" '{resource_type}.test_resource' but it is already registered by '{c1.name}'."
    )
    with pytest.raises(ResourceError, match=error_message):
        manager.add_resource(duplicate_resource)


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream, mocker: MockerFixture
) -> None:
    column = Column("c_1", ColumnCreatorAndRequirer(), [randomness_stream])
    randomness_stream._required_resources.append(column)
    manager.add_resource(column)
    manager.add_resource(randomness_stream)

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager,
) -> None:
    component = ColumnCreator()
    # Create a three-node cycle: A -> B -> C -> A
    resource_a = Resource("A", component, [])
    resource_b = Resource("B", component, [])
    resource_c = Resource("C", component, [])

    # Set up the cycle
    resource_a._required_resources.append(resource_b)
    resource_b._required_resources.append(resource_c)
    resource_c._required_resources.append(resource_a)

    manager.add_resource(resource_a)
    manager.add_resource(resource_b)
    manager.add_resource(resource_c)

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    component = ColumnCreator()
    resources = [Resource(f"resource_{i}", component, []) for i in range(10)]
    for i in range(10):
        resources[i]._required_resources.append(resources[(i + 1) % 10])
        manager.add_resource(resources[i])

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager) -> None:
    component = ColumnCreator()

    initializer = Initializer(0, component, lambda _: None, [])
    manager.add_resource(initializer)
    initializers = [initializer]

    for i in range(1, 10):
        initializer = Initializer(i, component, lambda _: None, [initializers[0]])
        initializers.insert(0, initializer)
        manager.add_resource(initializer)

    for i, resource in enumerate(manager.sorted_nodes):
        assert resource.resource_id == initializers[9 - i].resource_id


def test_resource_manager_sorted_nodes_acyclic(
    manager_with_resources: ResourceManager,
) -> None:

    nodes = [node.resource_id for node in manager_with_resources.sorted_nodes]

    assert nodes.index("generic_resource.A") < nodes.index("stream.B")
    assert nodes.index("generic_resource.A") < nodes.index("generic_resource.C")
    assert nodes.index("generic_resource.A") < nodes.index("generic_resource.D")

    assert nodes.index("stream.B") < nodes.index("generic_resource.D")
    assert nodes.index("generic_resource.C") < nodes.index("generic_resource.D")

    assert nodes.index("initializer.0.test_3.initialize_D") < nodes.index("column.D")
    assert nodes.index("stream.B") < nodes.index("initializer.0.test_3.initialize_D")
    assert nodes.index("generic_resource.C") < nodes.index(
        "initializer.0.test_3.initialize_D"
    )

    assert nodes.index("initializer.1.test_0.initialize_A") < nodes.index("column.A")

    assert nodes.index("stream.B") < nodes.index("initializer.2.test_4.initialize_nothing")


# TODO MIC-6839: Add tests for add_private_columns
# TODO MIC-6840: Add tests for get_graph


def test_get_population_initializers(
    manager_with_resources: ResourceManager, resource_producers: dict[int, ResourceProducer]
) -> None:
    initializers = manager_with_resources.get_population_initializers()

    assert len(initializers) == 3
    assert initializers[0] == resource_producers[0].initialize_A
    assert resource_producers[3].initialize_D in initializers
    assert resource_producers[4].initialize_nothing in initializers
