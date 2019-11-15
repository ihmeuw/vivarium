import pytest

from vivarium.framework.engine import SimulationContext, Builder
from vivarium.framework.artifact import ArtifactManager, ArtifactInterface
from vivarium.framework.event import EventManager, EventInterface
from vivarium.framework.lookup import LookupTableManager, LookupTableInterface
from vivarium.framework.components import OrderedComponentSet, ComponentManager, ComponentInterface
from vivarium.framework.metrics import Metrics
from vivarium.framework.population import PopulationManager, PopulationInterface
from vivarium.framework.randomness import RandomnessManager, RandomnessInterface
from vivarium.framework.resource import ResourceManager, ResourceInterface
from vivarium.framework.time import DateTimeClock, TimeInterface
from vivarium.framework.values import ValuesManager, ValuesInterface

from .components.mocks import MockComponentA, MockComponentB, Listener


def is_same_object_method(m1, m2):
    return m1.__func__ is m2.__func__ and m1.__self__ is m2.__self__


@pytest.fixture
def components():
    return [MockComponentA('gretchen', 'whimsy'),
            Listener('listener'),
            MockComponentB('spoon', 'antelope', 23)]


@pytest.fixture
def log(mocker):
    return mocker.patch('vivarium.framework.engine.logger')


def test_SimulationContext_init_default(components):
    sim = SimulationContext(components=components)

    assert isinstance(sim._component_manager, ComponentManager)
    assert isinstance(sim._clock, DateTimeClock)
    assert isinstance(sim._data, ArtifactManager)
    assert isinstance(sim._values, ValuesManager)
    assert isinstance(sim._events, EventManager)
    assert isinstance(sim._population, PopulationManager)
    assert isinstance(sim._tables, LookupTableManager)
    assert isinstance(sim._randomness, RandomnessManager)

    assert isinstance(sim._builder, Builder)
    assert sim._builder.configuration is sim.configuration
    assert isinstance(sim._builder.lookup, LookupTableInterface)
    assert sim._builder.lookup._manager is sim._tables
    assert isinstance(sim._builder.value, ValuesInterface)
    assert sim._builder.value._manager is sim._values
    assert isinstance(sim._builder.event, EventInterface)
    assert sim._builder.event._manager is sim._events
    assert isinstance(sim._builder.population, PopulationInterface)
    assert sim._builder.population._manager is sim._population
    assert isinstance(sim._builder.randomness, RandomnessInterface)
    assert sim._builder.randomness._manager is sim._randomness
    assert isinstance(sim._builder.resources, ResourceInterface)
    assert sim._builder.resources._manager is sim._resource
    assert isinstance(sim._builder.time, TimeInterface)
    assert sim._builder.time._manager is sim._clock
    assert isinstance(sim._builder.components, ComponentInterface)
    assert sim._builder.components._manager is sim._component_manager
    assert isinstance(sim._builder.data, ArtifactInterface)
    assert sim._builder.data._manager is sim._data

    # Ordering matters.
    managers = [sim._clock, sim._lifecycle, sim._resource, sim._values,
                sim._population, sim._randomness, sim._events,
                sim._tables, sim._data]
    assert sim._component_manager._managers == OrderedComponentSet(*managers)
    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, 'sub_components'):
            unpacked_components.extend(c.sub_components)
    assert list(sim._component_manager._components)[:-1] == unpacked_components
    assert isinstance(list(sim._component_manager._components)[-1], Metrics)


def test_SimulationContext_setup_default(base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if 'listener' in c.args][0]
    assert not listener.post_setup_called
    sim.setup()

    unpacked_components = []
    for c in components:
        unpacked_components.append(c)
        if hasattr(c, 'sub_components'):
            unpacked_components.extend(c.sub_components)
    unpacked_components.append(Metrics())

    for a, b in zip(sim._component_manager._components, unpacked_components):
        assert type(a) == type(b)
        if hasattr(a, 'args'):
            assert a.args == b.args

    assert is_same_object_method(sim.simulant_creator, sim._population._create_simulants)
    assert sim.time_step_events == ['time_step__prepare', 'time_step', 'time_step__cleanup', 'collect_metrics']
    for k in sim.time_step_emitters.keys():
        assert is_same_object_method(sim.time_step_emitters[k], sim._events._event_types[k].emit)

    assert is_same_object_method(sim.end_emitter, sim._events._event_types['simulation_end'].emit)

    assert listener.post_setup_called


def test_SimulationContext_initialize_simulants(base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    pop_size = sim.configuration.population.population_size
    current_time = sim._clock.time
    assert sim._population.get_population(True).empty
    sim.initialize_simulants()
    assert len(sim._population.get_population(True)) == pop_size
    assert sim._clock.time == current_time


def test_SimulationContext_step(log, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()

    current_time = sim._clock.time
    step_size = sim._clock.step_size

    listener = [c for c in components if 'listener' in c.args][0]

    assert not listener.time_step__prepare_called
    assert not listener.time_step_called
    assert not listener.time_step__cleanup_called
    assert not listener.collect_metrics_called

    sim.step()

    assert log.debug.called_once_with(current_time)
    assert listener.time_step__prepare_called
    assert listener.time_step_called
    assert listener.time_step__cleanup_called
    assert listener.collect_metrics_called

    assert sim._clock.time == current_time + step_size


def test_SimulationContext_finalize(base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if 'listener' in c.args][0]
    sim.setup()
    sim.initialize_simulants()
    sim.step()
    assert not listener.simulation_end_called
    sim.finalize()
    assert listener.simulation_end_called


def test_SimulationContext_report(base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    sim.run()
    sim.finalize()
    metrics = sim.report()
    assert metrics['test'] == len([c for c in sim._component_manager._components if isinstance(c, MockComponentB)])
