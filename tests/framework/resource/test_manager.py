from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Any

import pytest
import pytest_mock

from tests.helpers import ColumnCreator, ColumnCreatorAndRequirer, ColumnRequirer
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
def resource_producers() -> dict[int, ResourceProducer]:
    return {i: ResourceProducer(f"test_{i}") for i in range(5)}


@pytest.fixture
def manager_with_resources(
    manager: ResourceManager, resource_producers: dict[int, ResourceProducer]
) -> ResourceManager:
    stream = RandomnessStream(
        "B", lambda: datetime.now(), 1, IndexMap(), resource_producers[1]
    )
    pipeline = Pipeline("C", resource_producers[2])

    manager.add_resources(resource_producers[3], ["D"], [stream, pipeline])
    manager.add_resources(stream.component, [stream], ["A"])
    manager.add_resources(pipeline.component, [pipeline], ["A"])
    manager.add_resources(resource_producers[0], ["A"], [])
    manager.add_resources(resource_producers[4], [], [stream])
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

    def on_initialize_simulants(self, _simulant_data: SimulantData) -> None:
        pass


@pytest.mark.parametrize(
    "resource_class, init_args, type_string, is_initializer",
    [
        (Pipeline, ["foo"], "value", False),
        (ValueSource, [Pipeline("foo"), lambda: 1], "value_source", False),
        (ValueModifier, [Pipeline("foo"), lambda: 1], "value_modifier", False),
        (Column, ["foo"], "column", True),
        (NullResource, [1], "null", True),
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
    component = ColumnCreator()

    group = manager._get_resource_group(
        component, [resource_class(*init_args, component=component)], []
    )

    assert group.type == type_string
    assert group.names == [r.resource_id for r in group.resources.values()]
    assert not group.dependencies
    assert group.is_initialized == is_initializer
    assert group.initializer == component.on_initialize_simulants


def test_resource_manager_get_resource_group_null(manager: ResourceManager) -> None:
    component_1 = ColumnCreator()
    component_2 = ColumnCreatorAndRequirer()

    group_1 = manager._get_resource_group(component_1, [], [])
    group_2 = manager._get_resource_group(component_2, [], [])

    assert group_1.type == "null"
    assert group_1.names == ["null.0"]
    assert group_1.initializer == component_1.on_initialize_simulants
    assert not group_1.dependencies

    assert group_2.type == "null"
    assert group_2.names == ["null.1"]
    assert group_2.initializer == component_2.on_initialize_simulants
    assert not group_2.dependencies


def test_add_resource_wrong_component(manager: ResourceManager) -> None:
    resource = Pipeline("foo", ColumnCreatorAndRequirer())
    error_message = "All initialized resources must have the component 'column_creator'."
    with pytest.raises(ResourceError, match=error_message):
        manager.add_resources(ColumnCreator(), [resource], [])


@pytest.mark.parametrize(
    "resource_type, resource_creator",
    [
        ("column", lambda name, component: name),
        ("value", lambda name, component: Pipeline(name, component)),
    ],
)
def test_resource_manager_add_same_resource_twice(
    resource_type: str,
    resource_creator: Callable[[str, Component], Any],
    manager: ResourceManager,
) -> None:
    c1 = ColumnCreator()
    c2 = ColumnCreatorAndRequirer()
    r1 = [resource_creator(str(i), c1) for i in range(5)]
    r2 = [resource_creator(str(i), c2) for i in range(5, 10)] + [resource_creator("1", c2)]

    manager.add_resources(c1, r1, [])
    error_message = (
        f"Component '{c2.name}' is attempting to register resource"
        f" '{resource_type}.1' but it is already registered by '{c1.name}'."
    )
    with pytest.raises(ResourceError, match=error_message):
        manager.add_resources(c2, r2, [])


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    manager.add_resources(ColumnCreatorAndRequirer(), ["c_1"], [randomness_stream])
    manager.add_resources(randomness_stream.component, [randomness_stream], ["c_1"])

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream
) -> None:
    pipeline = Pipeline("some_pipeline", ColumnRequirer())

    manager.add_resources(ColumnCreatorAndRequirer(), ["c_1"], [randomness_stream])
    manager.add_resources(pipeline.component, [pipeline], ["c_1"])
    manager.add_resources(randomness_stream.component, [randomness_stream], [pipeline])

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    component = ColumnCreator()
    for i in range(10):
        manager.add_resources(component, [f"c_{i}"], [f"c_{i % 10}"])

    with pytest.raises(ResourceError, match="cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager) -> None:
    component = ColumnCreator()
    for i in range(9, 0, -1):
        manager.add_resources(component, [f"c_{i}"], [f"c_{i - 1}"])
    manager.add_resources(component, ["c_0"], [])

    for i, resource in enumerate(manager.sorted_nodes):
        assert str(resource) == f"(column.c_{i})"


def test_resource_manager_sorted_nodes_acyclic(
    manager_with_resources: ResourceManager,
) -> None:

    n = [str(node) for node in manager_with_resources.sorted_nodes]

    assert n.index("(column.A)") < n.index("(stream.B)")
    assert n.index("(column.A)") < n.index("(value.C)")
    assert n.index("(column.A)") < n.index("(column.D)")

    assert n.index("(stream.B)") < n.index("(column.D)")
    assert n.index("(value.C)") < n.index("(column.D)")

    assert n.index("(stream.B)") < n.index(f"(null.0)")


def test_get_population_initializers(
    manager_with_resources: ResourceManager, resource_producers: dict[int, ResourceProducer]
) -> None:
    initializers = manager_with_resources.get_population_initializers()

    assert len(initializers) == 3
    assert initializers[0] == resource_producers[0].on_initialize_simulants
    assert resource_producers[3].on_initialize_simulants in initializers
    assert resource_producers[4].on_initialize_simulants in initializers
