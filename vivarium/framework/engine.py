"""The engine."""
import gc
import logging
from pprint import pformat
from time import time

import pandas as pd

from vivarium.configuration import build_simulation_configuration
from .components import get_component_configuration_parser, get_component_manager, get_dataset_manager
from .builder import Builder
from .event import EventManager, Event
from .lookup import InterpolatedDataManager
from .population import PopulationManager
from .randomness import RandomnessManager
from .results_writer import get_results_writer
from .time import get_clock
from .values import ValuesManager, DynamicValueError

_log = logging.getLogger(__name__)


class SimulationContext:
    """context"""
    def __init__(self, simulation_config, component_config=None):
        self.configuration = simulation_config

        component_config_parser = get_component_configuration_parser(simulation_config)
        components = component_config_parser.get_components(component_config)
        self.component_manager = get_component_manager(simulation_config)
        self.component_manager.add_components(components)

        self.dataset_manager = get_dataset_manager(simulation_config)
        self.clock = get_clock(simulation_config)

        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = InterpolatedDataManager()
        self.randomness = RandomnessManager()

    def setup(self):
        builder = Builder(self)

        self.component_manager.add_managers(
            [self.clock, self.randomness, self.values, self.events, self.population, self.tables])
        self.component_manager.setup_components(builder)

        self.simulant_creator = builder.population.get_simulant_creator()

        # The order here matters.
        self.time_step_events = ['time_step__prepare', 'time_step', 'time_step__cleanup', 'collect_metrics']
        self.time_step_emitters = {k: builder.event.get_emitter(k) for k in self.time_step_events}

        self.end_emitter = builder.event.get_emitter('simulation_end')

        self.events.get_emitter('post_setup')(None)

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

    def __repr__(self):
        return "SimulationContext()"


def event_loop(simulation):
    simulation.initialize_simulants()

    while simulation.clock.time < simulation.clock.stop_time:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        simulation.step()

    simulation.finalize()


def setup_simulation(config):
    component_config = config.components
    simulation_config = config.configuration

    simulation_config.run_configuration.set_with_metadata('run_id', str(time()), layer='base')
    simulation = SimulationContext(simulation_config, component_config)
    simulation.setup()
    return simulation


def run_simulation(simulation):
    start = time()
    event_loop(simulation)
    try:
        metrics = simulation.values.get_value('metrics')
    except DynamicValueError:
        metrics = simulation.values.register_value_producer('metrics', source=lambda index: {})
    metrics = metrics(simulation.population.population.index)
    metrics['simulation_run_time'] = time() - start
    return metrics


def run(simulation):
    metrics = run_simulation(simulation)

    _log.debug(pformat(metrics))

    unused_config_keys = simulation.configuration.unused_keys()
    if unused_config_keys:
        _log.debug("Some configuration keys not used during run: %s", unused_config_keys)

    return metrics, simulation.population._population


def do_command(args):
    config = build_simulation_configuration(vars(args))
    simulation = setup_simulation(config)
    if args.command == 'run':
        results_writer = get_results_writer(config.configuration.run_configuration.results_directory,
                                            args.simulation_configuration)
        metrics, final_state = run(simulation)
        idx = pd.Index([config.configuration.randomness.random_seed], name='random_seed')
        output = pd.DataFrame(metrics, index=idx)
        results_writer.write_output(output, 'output.hdf')
        results_writer.write_output(final_state, 'final_state.hdf')
