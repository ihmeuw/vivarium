"""The engine."""
import gc
import logging
from pprint import pformat
from time import time

import pandas as pd

from vivarium.framework.configuration import build_model_specification
from .components import ComponentsInterface
from .event import EventManager, Event, EventsInterface
from .lookup import InterpolatedDataManager
from .metrics import Metrics
from .plugins import PluginManager
from .population import PopulationManager, PopulationInterface
from .randomness import RandomnessManager, RandomnessInterface
from .results_writer import get_results_writer
from .values import ValuesManager, ValuesInterface
from .time import TimeInterface

_log = logging.getLogger(__name__)


class SimulationContext:
    """context"""
    def __init__(self, configuration, components, plugin_manager=None):
        self.configuration = configuration
        self.plugin_manager = plugin_manager if plugin_manager else PluginManager(configuration)
        self.component_manager = self.plugin_manager.get_plugin('component_manager')
        self.component_manager.add_components(components)
        self.clock = self.plugin_manager.get_plugin('clock')

        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = InterpolatedDataManager()
        self.randomness = RandomnessManager()

        self.builder = Builder(self)
        self.builder.components = self.plugin_manager.get_plugin_interface('component_manager')
        self.builder.time = self.plugin_manager.get_plugin_interface('clock')
        for name, interface in self.plugin_manager.get_optional_interfaces().items():
            setattr(self.builder, name, interface)

        self.component_manager.add_managers(
            [self.clock, self.population, self.randomness, self.values, self.events, self.tables])
        self.component_manager.add_managers(list(self.plugin_manager.get_optional_controllers().values()))
        self.component_manager.add_components([Metrics()])

    def setup(self):
        self.component_manager.setup_components(self.builder)

        self.simulant_creator = self.builder.population.get_simulant_creator()

        # The order here matters.
        self.time_step_events = ['time_step__prepare', 'time_step', 'time_step__cleanup', 'collect_metrics']
        self.time_step_emitters = {k: self.builder.event.get_emitter(k) for k in self.time_step_events}
        self.end_emitter = self.builder.event.get_emitter('simulation_end')
        self.builder.event.get_emitter('post_setup')(None)

    def step(self):
        _log.debug(self.clock.time)
        for event in self.time_step_events:
            self.time_step_emitters[event](Event(self.population.population.index))
        self.clock.step_forward()

    def initialize_simulants(self):
        pop_params = self.configuration.population

        # Fencepost the creation of the initial population.
        self.clock.step_backward()
        population_size = pop_params.population_size
        self.simulant_creator(population_size)
        self.clock.step_forward()

    def finalize(self):
        self.end_emitter(Event(self.population.population.index))

    def report(self):
        return self.values.get_value('metrics')(self.population.population.index)

    def __repr__(self):
        return "SimulationContext()"


class Builder:
    """Toolbox for constructing and configuring simulation components."""

    def __init__(self, context):
        self.configuration = context.configuration
        self.lookup = context.tables.build_table
        self.value = ValuesInterface(context.values)
        self.event = EventsInterface(context.events)
        self.population = PopulationInterface(context.population)
        self.randomness = RandomnessInterface(context.randomness)

        # These set in SimulationContext.setup()
        self.time = None  # type: TimeInterface
        self.components = None  # type: ComponentsInterface

    def __repr__(self):
        return "Builder()"


def run_simulation(model_specification_file, results_directory):
    results_writer = get_results_writer(results_directory, model_specification_file)

    model_specification = build_model_specification(model_specification_file)
    model_specification.configuration.output_data.update(
        {'results_directory': results_writer.results_root}, layer='override', source='command_line')

    simulation = setup_simulation(model_specification)
    metrics, final_state = run(simulation)

    _log.debug(pformat(metrics))
    unused_config_keys = simulation.configuration.unused_keys()
    if unused_config_keys:
        _log.debug("Some configuration keys not used during run: %s", unused_config_keys)

    metrics = pd.DataFrame(metrics)
    results_writer.write_output(metrics, 'output.hdf')
    results_writer.write_output(final_state, 'final_state.hdf')


def setup_simulation(model_specification):
    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config, simulation_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    simulation = SimulationContext(simulation_config, components, plugin_manager)
    simulation.setup()
    return simulation


def run(simulation):
    start = time()
    simulation.initialize_simulants()

    while simulation.clock.time < simulation.clock.stop_time:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        simulation.step()

    simulation.finalize()
    metrics = simulation.report()
    metrics['simulation_run_time'] = time() - start
    return metrics, simulation.population.population
