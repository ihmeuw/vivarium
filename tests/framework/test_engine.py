from typing import List

import pytest

from vivarium import Component
from vivarium.framework.artifact import ArtifactInterface, ArtifactManager
from vivarium.framework.components import (
    ComponentConfigError,
    ComponentInterface,
    ComponentManager,
    OrderedComponentSet,
)
from vivarium.framework.engine import Builder
from vivarium.framework.engine import SimulationContext as SimulationContext_
from vivarium.framework.event import EventInterface, EventManager
from vivarium.framework.lifecycle import LifeCycleInterface, LifeCycleManager
from vivarium.framework.logging import LoggingInterface, LoggingManager
from vivarium.framework.lookup import LookupTableInterface, LookupTableManager
from vivarium.framework.metrics import Metrics
from vivarium.framework.population import PopulationInterface, PopulationManager
from vivarium.framework.randomness import RandomnessInterface, RandomnessManager
from vivarium.framework.resource import ResourceInterface, ResourceManager
from vivarium.framework.results import ResultsInterface, ResultsManager
from vivarium.framework.time import DateTimeClock, TimeInterface
from vivarium.framework.values import ValuesInterface, ValuesManager

from .components.mocks import Listener, MockComponentA, MockComponentB


def is_same_object_method(m1, m2):
    return m1.__func__ is m2.__func__ and m1.__self__ is m2.__self__


@pytest.fixture()
def SimulationContext():
    yield SimulationContext_
    SimulationContext_._clear_context_cache()


@pytest.fixture
def components():
    return [
        MockComponentA("gretchen", "whimsy"),
        Listener("listener"),
        MockComponentB("spoon", "antelope", 23),
    ]


@pytest.fixture
def log(mocker):
    return mocker.patch("vivarium.framework.logging.manager.logger")


@pytest.mark.xfail
def test_simulation_with_non_components(SimulationContext, components: List[Component]):
    class NonComponent:
        def __init__(self):
            self.name = "non_component"

    with pytest.raises(ComponentConfigError):
        # noinspection PyTypeChecker
        SimulationContext(components=components + [NonComponent()])


def test_SimulationContext_get_sim_name(SimulationContext):
    assert SimulationContext._created_simulation_contexts == set()

    assert SimulationContext._get_context_name(None) == "simulation_1"
    assert SimulationContext._get_context_name("foo") == "foo"

    assert SimulationContext._created_simulation_contexts == {"simulation_1", "foo"}


def test_SimulationContext_init_default(SimulationContext, components):
    sim = SimulationContext(components=components)

    assert isinstance(sim._logging, LoggingManager)
    assert isinstance(sim._lifecycle, LifeCycleManager)
    assert isinstance(sim._component_manager, ComponentManager)
    assert isinstance(sim._clock, DateTimeClock)
    assert isinstance(sim._values, ValuesManager)
    assert isinstance(sim._events, EventManager)
    assert isinstance(sim._population, PopulationManager)
    assert isinstance(sim._resource, ResourceManager)
    assert isinstance(sim._results, ResultsManager)
    assert isinstance(sim._tables, LookupTableManager)
    assert isinstance(sim._randomness, RandomnessManager)
    assert isinstance(sim._data, ArtifactManager)

    assert isinstance(sim._builder, Builder)
    assert sim._builder.configuration is sim.configuration

    assert isinstance(sim._builder.logging, LoggingInterface)
    assert sim._builder.logging._manager is sim._logging
    assert isinstance(sim._builder.lookup, LookupTableInterface)
    assert sim._builder.lookup._manager is sim._tables
    assert isinstance(sim._builder.value, ValuesInterface)
    assert sim._builder.value._manager is sim._values
    assert isinstance(sim._builder.event, EventInterface)
    assert sim._builder.event._manager is sim._events
    assert isinstance(sim._builder.population, PopulationInterface)
    assert sim._builder.population._manager is sim._population
    assert isinstance(sim._builder.resources, ResourceInterface)
    assert sim._builder.resources._manager is sim._resource
    assert isinstance(sim._builder.results, ResultsInterface)
    assert sim._builder.results._manager is sim._results
    assert isinstance(sim._builder.randomness, RandomnessInterface)
    assert sim._builder.randomness._manager is sim._randomness
    assert isinstance(sim._builder.time, TimeInterface)
    assert sim._builder.time._manager is sim._clock
    assert isinstance(sim._builder.components, ComponentInterface)
    assert sim._builder.components._manager is sim._component_manager
    assert isinstance(sim._builder.lifecycle, LifeCycleInterface)
    assert sim._builder.lifecycle._manager is sim._lifecycle
    assert isinstance(sim._builder.data, ArtifactInterface)
    assert sim._builder.data._manager is sim._data

    # Ordering matters.
    managers = [
        sim._logging,
        sim._clock,
        sim._lifecycle,
        sim._resource,
        sim._values,
        sim._population,
        sim._randomness,
        sim._events,
        sim._tables,
        sim._data,
        sim._results,
    ]
    assert sim._component_manager._managers == OrderedComponentSet(*managers)
    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, "sub_components"):
            unpacked_components.extend(c.sub_components)
    assert list(sim._component_manager._components)[:-1] == unpacked_components
    assert isinstance(list(sim._component_manager._components)[-1], Metrics)


def test_SimulationContext_name_management(SimulationContext):
    assert SimulationContext._created_simulation_contexts == set()

    sim1 = SimulationContext()
    assert sim1._name == "simulation_1"
    assert SimulationContext._created_simulation_contexts == {"simulation_1"}

    sim2 = SimulationContext(sim_name="foo")
    assert sim2._name == "foo"
    assert SimulationContext._created_simulation_contexts == {"simulation_1", "foo"}

    sim3 = SimulationContext()
    assert sim3._name == "simulation_3"
    assert SimulationContext._created_simulation_contexts == {
        "simulation_1",
        "foo",
        "simulation_3",
    }


def test_SimulationContext_setup_default(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    assert not listener.post_setup_called
    sim.setup()

    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, "sub_components"):
            unpacked_components.extend(c.sub_components)
    unpacked_components.append(Metrics())

    for a, b in zip(sim._component_manager._components, unpacked_components):
        assert type(a) == type(b)
        if hasattr(a, "args"):
            assert a.args == b.args

    assert is_same_object_method(sim.simulant_creator, sim._population._create_simulants)
    assert sim.time_step_events == [
        "time_step__prepare",
        "time_step",
        "time_step__cleanup",
        "collect_metrics",
    ]
    for k in sim.time_step_emitters.keys():
        assert is_same_object_method(
            sim.time_step_emitters[k], sim._events._event_types[k].emit
        )

    assert is_same_object_method(
        sim.end_emitter, sim._events._event_types["simulation_end"].emit
    )

    assert listener.post_setup_called


def test_SimulationContext_initialize_simulants(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    pop_size = sim.configuration.population.population_size
    current_time = sim._clock.time
    assert sim._population.get_population(True).empty
    sim.initialize_simulants()
    assert len(sim._population.get_population(True)) == pop_size
    assert sim._clock.time == current_time


def test_SimulationContext_step(SimulationContext, log, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()

    current_time = sim._clock.time
    step_size = sim._clock.step_size

    listener = [c for c in components if "listener" in c.args][0]

    assert not listener.time_step_prepare_called
    assert not listener.time_step_called
    assert not listener.time_step_cleanup_called
    assert not listener.collect_metrics_called

    sim.step()

    assert log.debug.called_once_with(current_time)
    assert listener.time_step_prepare_called
    assert listener.time_step_called
    assert listener.time_step_cleanup_called
    assert listener.collect_metrics_called

    assert sim._clock.time == current_time + step_size


def test_SimulationContext_finalize(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if "listener" in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    sim.step()
    assert not listener.simulation_end_called
    sim.finalize()
    assert listener.simulation_end_called


def test_SimulationContext_report(SimulationContext, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    sim.run()
    sim.finalize()
    metrics = sim.report()
    assert metrics["test"] == len(
        [c for c in sim._component_manager._components if isinstance(c, MockComponentB)]
    )
