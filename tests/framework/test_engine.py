import pytest

from vivarium.framework.engine import SimulationContext, Builder, setup_simulation, run, run_simulation
from vivarium.framework.event import EventManager, EventInterface
from vivarium.framework.lookup import InterpolatedDataManager, LookupTableInterface
from vivarium.framework.components import ComponentManager, ComponentInterface
from vivarium.framework.metrics import Metrics
from vivarium.framework.plugins import PluginManager
from vivarium.framework.population import PopulationManager, PopulationInterface
from vivarium.framework.randomness import RandomnessManager, RandomnessInterface
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
    return mocker.patch('vivarium.framework.engine._log')


def test_SimulationContext_init_default(base_config, components):
    sim = SimulationContext(base_config, components)

    assert sim.configuration == base_config
    assert isinstance(sim.component_manager, ComponentManager)
    assert isinstance(sim.clock, DateTimeClock)
    assert isinstance(sim.values, ValuesManager)
    assert isinstance(sim.events, EventManager)
    assert isinstance(sim.population, PopulationManager)
    assert isinstance(sim.tables, InterpolatedDataManager)
    assert isinstance(sim.randomness, RandomnessManager)

    assert isinstance(sim.builder, Builder)
    assert sim.builder.configuration is sim.configuration
    assert isinstance(sim.builder.lookup, LookupTableInterface)
    assert sim.builder.lookup._lookup_table_manager is sim.tables
    assert isinstance(sim.builder.value, ValuesInterface)
    assert sim.builder.value._value_manager is sim.values
    assert isinstance(sim.builder.event, EventInterface)
    assert sim.builder.event._event_manager is sim.events
    assert isinstance(sim.builder.population, PopulationInterface)
    assert sim.builder.population._population_manager is sim.population
    assert isinstance(sim.builder.randomness, RandomnessInterface)
    assert sim.builder.randomness._randomness_manager is sim.randomness
    assert isinstance(sim.builder.time, TimeInterface)
    assert sim.builder.time._clock is sim.clock
    assert isinstance(sim.builder.components, ComponentInterface)
    assert sim.builder.components._component_manager is sim.component_manager

    # Ordering matters.
    managers = [sim.clock, sim.population, sim.randomness, sim.values, sim.events, sim.tables]
    assert sim.component_manager._managers == managers
    assert sim.component_manager._components[:-1] == components
    assert isinstance(sim.component_manager._components[-1], Metrics)


def test_SimulationContext_init_custom(base_config, components):
    beehive = MockComponentA('COVERED IN BEES')
    beekeeper = MockComponentB("Gets the honey")

    def plugin_interfaces_mock():
        return {'beehive': beehive}

    def plugin_controllers_mock():
        return {'beekeeper': beekeeper}

    plugin_manager = PluginManager()
    plugin_manager.get_optional_interfaces = plugin_interfaces_mock
    plugin_manager.get_optional_controllers = plugin_controllers_mock
    sim = SimulationContext(base_config, components, plugin_manager)

    assert sim.configuration == base_config
    assert isinstance(sim.component_manager, ComponentManager)
    assert isinstance(sim.clock, DateTimeClock)
    assert isinstance(sim.values, ValuesManager)
    assert isinstance(sim.events, EventManager)
    assert isinstance(sim.population, PopulationManager)
    assert isinstance(sim.tables, InterpolatedDataManager)
    assert isinstance(sim.randomness, RandomnessManager)

    assert isinstance(sim.builder, Builder)
    assert sim.builder.configuration is sim.configuration
    assert isinstance(sim.builder.lookup, LookupTableInterface)
    assert sim.builder.lookup._lookup_table_manager is sim.tables
    assert isinstance(sim.builder.value, ValuesInterface)
    assert sim.builder.value._value_manager is sim.values
    assert isinstance(sim.builder.event, EventInterface)
    assert sim.builder.event._event_manager is sim.events
    assert isinstance(sim.builder.population, PopulationInterface)
    assert sim.builder.population._population_manager is sim.population
    assert isinstance(sim.builder.randomness, RandomnessInterface)
    assert sim.builder.randomness._randomness_manager is sim.randomness
    assert isinstance(sim.builder.time, TimeInterface)
    assert sim.builder.time._clock is sim.clock
    assert isinstance(sim.builder.components, ComponentInterface)
    assert sim.builder.components._component_manager is sim.component_manager
    assert sim.builder.beehive == beehive

    # Ordering matters.
    managers = [sim.clock, sim.population, sim.randomness, sim.values, sim.events, sim.tables, beekeeper]
    assert sim.component_manager._managers == managers
    assert sim.component_manager._components[:-1] == components
    assert isinstance(sim.component_manager._components[-1], Metrics)


def test_SimulationContext_setup_default(base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if 'listener' in c.args][0]
    assert not listener.post_setup_called
    sim.setup()

    def unpack(component):
        if isinstance(component, MockComponentB) and len(component.args) > 1:
            return [component] + [MockComponentB(arg) for arg in component.args]
        return [component]
    unpacked_components = [c for component in components for c in unpack(component)]
    unpacked_components.insert(len(components), Metrics())
    for a, b in zip(sim.component_manager._components, unpacked_components):
        assert type(a) == type(b)
        if hasattr(a, 'args'):
            assert a.args == b.args

    assert is_same_object_method(sim.simulant_creator, sim.population._create_simulants)
    assert sim.time_step_events == ['time_step__prepare', 'time_step', 'time_step__cleanup', 'collect_metrics']
    for k in sim.time_step_emitters.keys():
        assert is_same_object_method(sim.time_step_emitters[k], sim.events._event_types[k].emit)

    assert is_same_object_method(sim.end_emitter, sim.events._event_types['simulation_end'].emit)

    assert listener.post_setup_called


def test_SimulationContext_step(log, base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()

    current_time = sim.clock.time
    step_size = sim.clock.step_size

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

    assert sim.clock.time == current_time + step_size


def test_SimulationContext_initialize_simulants(base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    pop_size = sim.configuration.population.population_size
    current_time = sim.clock.time
    assert sim.population.population.empty
    sim.initialize_simulants()
    assert len(sim.population.population) == pop_size
    assert sim.clock.time == current_time


def test_SimulationContext_finalize(base_config, components):
    sim = SimulationContext(base_config, components)
    listener = [c for c in components if 'listener' in c.args][0]
    sim.setup()
    assert not listener.simulation_end_called
    sim.finalize()
    assert listener.simulation_end_called


def test_SimulationContext_report(base_config, components):
    sim = SimulationContext(base_config, components)
    sim.setup()
    sim.initialize_simulants()
    metrics = sim.report()
    assert metrics['test'] == len([c for c in sim.component_manager._components if isinstance(c, MockComponentB)])


def test_setup_simulation(model_specification, mocker):
    plugin_manager_constructor_mock = mocker.patch('vivarium.framework.engine.PluginManager')
    plugin_manager_mock = mocker.MagicMock()
    plugin_manager_constructor_mock.return_value = plugin_manager_mock
    context_constructor_mock = mocker.patch('vivarium.framework.engine.SimulationContext')
    context_mock = mocker.MagicMock()
    context_constructor_mock.return_value = context_mock
    component_config_parser = mocker.MagicMock()
    component_config_parser.get_components.return_value = ['test']
    plugin_manager_mock.get_plugin.return_value = component_config_parser

    setup_simulation(model_specification)

    plugin_manager_constructor_mock.assert_called_once_with(model_specification.plugins)
    plugin_manager_mock.get_plugin.assert_called_once_with('component_configuration_parser')
    component_config_parser.get_components.assert_called_once_with(model_specification.components)
    context_constructor_mock.assert_called_once_with(model_specification.configuration, ['test'], plugin_manager_mock)
    context_mock.setup.assert_called_once()


def test_run(mocker):
    sim_mock = mocker.Mock()
    sim_mock.clock.time = 0
    sim_mock.clock.stop_time = 10

    def step():
        sim_mock.clock.time += 1

    sim_mock.step.side_effect = step
    sim_mock.report.return_value = {}
    pop = list(range(10))
    sim_mock.population.population = pop

    metrics, population = run(sim_mock)

    sim_mock.initialize_simulants.assert_called_once()
    assert sim_mock.step.call_count == sim_mock.clock.stop_time
    assert sim_mock.clock.time == sim_mock.clock.stop_time
    sim_mock.finalize.assert_called_once()
    sim_mock.report.assert_called_once()
    assert list(metrics.keys()) == ['simulation_run_time']
    assert metrics['simulation_run_time'] > 0
    assert pop is population


def test_run_simulation(model_specification, mocker):
    get_results_writer_mock = mocker.patch('vivarium.framework.engine.get_results_writer')
    results_writer_mock = mocker.Mock()
    results_writer_mock.results_root = 'test_dir'
    get_results_writer_mock.return_value = results_writer_mock

    build_model_spec_mock = mocker.patch('vivarium.framework.engine.build_model_specification')
    build_model_spec_mock.return_value = model_specification

    setup_simulation_mock = mocker.patch('vivarium.framework.engine.setup_simulation')
    sim_mock = mocker.Mock()
    unused_keys = ['not', 'used', 'at', 'all']
    sim_mock.configuration.unused_keys.return_value = unused_keys
    setup_simulation_mock.return_value = sim_mock

    run_mock = mocker.patch('vivarium.framework.engine.run')
    metrics = {'test_metrics': 'test'}
    final_state = 'the_final_state'
    run_mock.return_value = metrics, final_state

    log_mock = mocker.patch('vivarium.framework.engine._log')
    pformat_mock = mocker.patch('vivarium.framework.engine.pformat')
    pformat_mock.side_effect = lambda x: x

    pandas_mock = mocker.patch('vivarium.framework.engine.pd')
    pandas_mock.DataFrame.side_effect = lambda x, index: x

    model_spec_path = '/this/is/a/test.yaml'
    results_directory = 'test_dir'

    assert 'output_data' not in model_specification.configuration

    run_simulation(model_spec_path, results_directory)

    get_results_writer_mock.assert_called_once_with(results_directory, model_spec_path)

    build_model_spec_mock.assert_called_once_with(model_spec_path)
    assert model_specification.configuration.output_data.results_directory == results_directory

    setup_simulation_mock.assert_called_once_with(model_specification)
    run_mock.assert_called_once_with(sim_mock)

    pformat_mock.assert_called_once_with(metrics)
    log_mock.debug.assert_has_calls([mocker.call(metrics),
                                     mocker.call("Some configuration keys not used during run: %s", unused_keys)])

    results_writer_mock.write_output.assert_has_calls([mocker.call(metrics, 'output.hdf'),
                                                       mocker.call(final_state, 'final_state.hdf')])
