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
from vivarium.framework.resource.resource import Column, NullResource
from vivarium.framework.values import AttributePipeline, Pipeline, ValueModifier, ValueSource


@pytest.fixture
def manager(mocker: MockerFixture) -> ResourceManager:
    manager = ResourceManager()
    manager.logger = mocker.Mock()
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
    stream = RandomnessStream(
        key="B",
        clock=lambda: datetime.now(),
        seed=1,
        index_map=IndexMap(),
        component=resource_producers[1],
    )
    pipeline = Pipeline("C", resource_producers[2])
    A_component = resource_producers[0]
    attribute_A = AttributePipeline("A", A_component)
    mocker.patch.object(A_component, "initialize_A", create=True)
    D_component = resource_producers[3]
    attribute_D = AttributePipeline("D", D_component)
    mocker.patch.object(D_component, "initialize_D", create=True)
    null_resource_component = resource_producers[4]
    mocker.patch.object(null_resource_component, "on_initialize_simulants", create=True)

    manager.add_resources(
        component=D_component,
        initializer=None,
        resources=attribute_D,
        dependencies=[stream, pipeline],
    )
    # Add the private column resource
    manager.add_resources(
        component=D_component,
        initializer=D_component.initialize_D,
        resources=[Column("D", D_component)],
        dependencies=[stream, pipeline],
    )

    stream_component = stream.component
    assert isinstance(stream_component, Component)
    manager.add_resources(
        component=stream_component, initializer=None, resources=stream, dependencies=["A"]
    )

    pipeline_component = pipeline.component
    assert isinstance(pipeline_component, Component)
    manager.add_resources(
        component=pipeline_component, initializer=None, resources=pipeline, dependencies=["A"]
    )

    manager.add_resources(
        component=A_component, initializer=None, resources=attribute_A, dependencies=[]
    )
    # Add the private column resource
    manager.add_resources(
        component=A_component,
        initializer=A_component.initialize_A,
        resources=[Column("A", A_component)],
        dependencies=[],
    )

    manager.add_resources(
        component=null_resource_component,
        initializer=null_resource_component.on_initialize_simulants,
        resources=[],
        dependencies=[stream],
    )

    # Call each resource group's on_post_setup to finalize dependencies
    attribute_pipelines = {"A": attribute_A, "D": attribute_D}
    for rg in manager._resource_group_map.values():
        rg.set_dependencies(attribute_pipelines)

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

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        pass


@pytest.mark.parametrize(
    "resource_class, init_args, type_string",
    [
        (Pipeline, ["foo"], "value"),
        (AttributePipeline, ["foo"], "attribute"),
        (ValueSource, [Pipeline("foo"), lambda: 1], "value_source"),
        (ValueModifier, [Pipeline("foo"), lambda: 1], "value_modifier"),
        (Column, ["foo"], "column"),
        (NullResource, [1], "null"),
    ],
    ids=lambda x: [x.__name__ if isinstance(x, type) else x],
)
@pytest.mark.parametrize("initialized", [True, False])
def test_resource_manager_get_resource_group(
    resource_class: type,
    init_args: list[Any],
    type_string: str,
    initialized: bool,
    manager: ResourceManager,
) -> None:
    component = ColumnCreator()

    group = manager._get_resource_group(
        component=component,
        initializer=component.on_initialize_simulants if initialized else None,
        resources=[resource_class(*init_args, component=component)],
        dependencies=[],
    )

    assert group.type == type_string
    assert group.names == [r.resource_id for r in group.resources.values()]
    assert not group.dependencies
    assert group.is_initialized == initialized
    assert group.initializer == (component.on_initialize_simulants if initialized else None)


def test_resource_manager_get_resource_group_null(manager: ResourceManager) -> None:
    component_1 = ColumnCreator()
    component_2 = ColumnCreatorAndRequirer()

    group_1 = manager._get_resource_group(
        component=component_1,
        initializer=component_1.on_initialize_simulants,
        resources=[],
        dependencies=[],
    )
    group_2 = manager._get_resource_group(
        component=component_2,
        initializer=component_2.on_initialize_simulants,
        resources=[],
        dependencies=[],
    )

    assert group_1.type == "null"
    assert group_1.names == ["null.0"]
    assert group_1.initializer == component_1.on_initialize_simulants
    assert not group_1.dependencies

    assert group_2.type == "null"
    assert group_2.names == ["null.1"]
    assert group_2.initializer == component_2.on_initialize_simulants
    assert not group_2.dependencies


def test_get_resource_group_multiple_initializers(manager: ResourceManager) -> None:
    class SomeComponent(Component):
        def initializer_1(self, pop_data: SimulantData) -> None:
            pass

        def initializer_2(self, pop_data: SimulantData) -> None:
            pass

    component = SomeComponent()

    group = manager._get_resource_group(
        component=component,
        initializer=component.initializer_1,
        resources=[Column("foo", component), Column("bar", component)],
        dependencies=[],
    )

    assert group.type == "column"
    assert group.names == ["column.foo", "column.bar"]
    assert group.initializer == component.initializer_1

    # Create another group with the same resources but a different initializer
    group2 = manager._get_resource_group(
        component=component,
        initializer=component.initializer_2,
        resources=Resource("test", "baz", component),
        dependencies=[],
    )

    assert group2.type == "test"
    assert group2.names == ["test.baz"]
    assert group2.initializer == component.initializer_2


def test_add_resource_wrong_component(manager: ResourceManager) -> None:
    resource = Pipeline("foo", ColumnCreatorAndRequirer())
    error_message = "All initialized resources in this resource group must have the component 'column_creator'."
    component = ColumnCreator()
    with pytest.raises(ResourceError, match=error_message):
        manager.add_resources(
            component=component,
            initializer=component.on_initialize_simulants,
            resources=resource,
            dependencies=[],
        )


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
    r1 = [resource_creator(str(i), c1) for i in range(5)]
    r2 = [resource_creator(str(i), c2) for i in range(5, 10)] + [resource_creator("1", c2)]
    manager.add_resources(
        component=c1, initializer=c1.on_initialize_simulants, resources=r1, dependencies=[]
    )
    error_message = (
        f"Component '{c2.name}' is attempting to register resource"
        f" '{resource_type}.1' but it is already registered by '{c1.name}'."
    )
    with pytest.raises(ResourceError, match=error_message):
        manager.add_resources(
            component=c2,
            initializer=c2.on_initialize_simulants,
            resources=r2,
            dependencies=[],
        )


def test_resource_manager_sorted_nodes_two_node_cycle(
    manager: ResourceManager, randomness_stream: RandomnessStream, mocker: MockerFixture
) -> None:
    component = ColumnCreatorAndRequirer()
    column = Column("c_1", mocker.Mock())
    manager.add_resources(
        component=component,
        initializer=component.on_initialize_simulants,
        resources=[column],
        dependencies=[randomness_stream],
    )
    manager.add_resources(
        component=randomness_stream.component,
        initializer=None,
        resources=randomness_stream,
        dependencies=[column],
    )

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle(
    manager: ResourceManager,
    randomness_stream: RandomnessStream,
    mocker: MockerFixture,
) -> None:
    pipeline = Pipeline("some_pipeline", mocker.Mock())
    component = ColumnCreatorAndRequirer()
    column = Column("c_1", component)
    manager.add_resources(
        component=component,
        initializer=component.on_initialize_simulants,
        resources=[column],
        dependencies=[randomness_stream],
    )
    manager.add_resources(
        component=pipeline.component,
        initializer=None,
        resources=pipeline,
        dependencies=[column],
    )
    manager.add_resources(
        component=randomness_stream.component,
        initializer=None,
        resources=randomness_stream,
        dependencies=[pipeline],
    )

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle(manager: ResourceManager) -> None:
    component = ColumnCreator()
    for i in range(10):
        resource = Resource("test", f"resource{i}", component)
        dependency = Resource("test", f"resource{(i + 1) % 10}", component)
        manager.add_resources(
            component=component,
            initializer=None,
            resources=resource,
            dependencies=[dependency],
        )

    with pytest.raises(ResourceError, match="The resource pool contains at least one cycle"):
        _ = manager.sorted_nodes


def test_large_dependency_chain(manager: ResourceManager, mocker: MockerFixture) -> None:
    component = ColumnCreator()
    for i in range(9, 0, -1):
        manager.add_resources(
            component=component,
            initializer=component.on_initialize_simulants,
            resources=AttributePipeline(f"c_{i}", component),
            dependencies=[AttributePipeline(f"c_{i-1}", mocker.Mock())],
        )
    manager.add_resources(
        component=component,
        initializer=component.on_initialize_simulants,
        resources=AttributePipeline("c_0", component),
        dependencies=[],
    )

    for i, resource in enumerate(manager.sorted_nodes):
        assert str(resource) == f"(attribute.c_{i})"


def test_resource_manager_sorted_nodes_acyclic(
    manager_with_resources: ResourceManager,
) -> None:

    nodes = [str(node) for node in manager_with_resources.sorted_nodes]

    assert nodes.index("(attribute.A)") < nodes.index("(stream.B)")
    assert nodes.index("(attribute.A)") < nodes.index("(value.C)")
    assert nodes.index("(attribute.A)") < nodes.index("(attribute.D)")

    assert nodes.index("(stream.B)") < nodes.index("(attribute.D)")
    assert nodes.index("(value.C)") < nodes.index("(attribute.D)")

    assert nodes.index("(stream.B)") < nodes.index(f"(null.0)")


def test_get_population_initializers(
    manager_with_resources: ResourceManager, resource_producers: dict[int, ResourceProducer]
) -> None:
    initializers = manager_with_resources.get_population_initializers()

    assert len(initializers) == 3
    assert initializers[0] == resource_producers[0].initialize_A
    assert resource_producers[3].initialize_D in initializers
    assert resource_producers[4].on_initialize_simulants in initializers
