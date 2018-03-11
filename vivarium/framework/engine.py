"""The engine."""
import argparse
import gc
import logging
from bdb import BdbQuit
from collections import namedtuple
from pprint import pformat, pprint
from time import time

import pandas as pd
import yaml

from vivarium.configuration import build_simulation_configuration

from .components import load_component_manager
from .event import EventManager, Event
from .lookup import InterpolatedDataManager
from .population import PopulationManager
from .randomness import RandomnessManager
from .results_writer import get_results_writer
from .time import get_clock
from .util import collapse_nested_dict
from .values import ValuesManager, DynamicValueError

_log = logging.getLogger(__name__)


class SimulationContext:
    """context"""
    def __init__(self, component_manager, configuration):
        self.component_manager = component_manager
        self.configuration = configuration
        self.clock = get_clock(self.configuration.vivarium.clock)
        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = InterpolatedDataManager()
        self.randomness = RandomnessManager()
        self.current_time = None
        self.step_size = pd.Timedelta(0, unit='D')

    def setup(self):
        builder = Builder(self)

        self.component_manager.add_components(
            [self.values, self.events, self.population, self.tables, self.randomness, self.clock])
        self.component_manager.load_components_from_config()
        self.component_manager.setup_components(builder)

        self.simulant_creator = builder.population.get_simulant_creator()
        # The order here matters.
        self.time_step_events = ['time_step__prepare', 'time_step', 'time_step__cleanup', 'collect_metrics']
        self.time_step_emitters = {k: builder.event.get_emitter(k) for k in self.time_step_events}

        self.end_emitter = builder.event.get_emitter('simulation_end')

        self.events.get_emitter('post_setup')(None)

    def step(self):
        _log.debug(self.current_time)
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
        return "SimulationContext(current_time={}, step_size={})".format(self.current_time, self.step_size)


class Builder:
    """Useful tools for constructing and configuring simulation components."""
    def __init__(self, context: SimulationContext):
        self.configuration = context.configuration

        self.lookup = context.tables.build_table

        _time = namedtuple('Time', ['clock', 'step_size'])
        self.time = _time(lambda: context.clock.time,
                          lambda: context.clock.step_size)

        _value = namedtuple('Value', ['register_value_producer', 'register_rate_producer', 'register_value_modifier'])
        self.value = _value(context.values.register_value_producer,
                            context.values.register_rate_producer,
                            context.values.register_value_modifier)

        _event = namedtuple('Event', ['get_emitter', 'register_listener'])
        self.event = _event(context.events.get_emitter, context.events.register_listener)

        _population = namedtuple('Population', ['get_view', 'get_simulant_creator', 'initializes_simulants'])
        self.population = _population(context.population.get_view,
                                      context.population.get_simulant_creator,
                                      context.population.register_simulant_initializer)

        _randomness = namedtuple('Randomness', ['get_stream', 'register_simulants'])
        self.randomness = _randomness(context.randomness.get_randomness_stream,
                                      context.randomness.register_simulants)

    def __repr__(self):
        return "Builder()"





def event_loop(simulation):
    simulation.initialize_simulants()

    sim_params = simulation.configuration.simulation_parameters
    stop = _get_time('end', sim_params)

    while simulation.current_time < stop:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        simulation.step()

    simulation.finalize()


def setup_simulation(component_manager, config):
    config.run_configuration.set_with_metadata('run_id', str(time()), layer='base')
    config.run_configuration.set_with_metadata('run_key',
                                               {'draw': config.run_configuration.input_draw_number}, layer='base')
    component_manager.add_components([event_loop])
    simulation = SimulationContext(component_manager, config)
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

    for name, value in collapse_nested_dict(simulation.configuration.run_configuration.run_key.to_dict()):
        metrics[name] = value
    _log.debug(pformat(metrics))

    unused_config_keys = simulation.configuration.unused_keys()
    if unused_config_keys:
        _log.debug("Some configuration keys not used during run: %s", unused_config_keys)

    return metrics, simulation.population._population


def do_command(args):
    config = build_simulation_configuration(vars(args))
    component_manager = load_component_manager(config)
    if args.command == 'run':
        simulation = setup_simulation(component_manager, config)
        results_writer = get_results_writer(config.run_configuration.results_directory, args.simulation_configuration)
        metrics, final_state = run(simulation)
        idx = pd.MultiIndex.from_tuples([(config.run_configuration.input_draw_number,
                                          config.run_configuration.model_draw_number)],
                                        names=['input_draw_number', 'model_draw_number'])
        output = pd.DataFrame(metrics, index=idx)
        results_writer.write_output(output, 'output.hdf')
        results_writer.write_output(final_state, 'final_state.hdf')
    elif args.command == 'list_datasets':
        component_manager.load_components_from_config()
        pprint(yaml.dump(list(component_manager.dataset_manager.datasets_loaded)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'list_datasets'])
    parser.add_argument('simulation_configuration', nargs='?', default=None, type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--random_seed', '-s', type=int, default=0, help="Seed for random number generation")
    parser.add_argument('--results_path', '-o', type=str, default=None, help='Output directory to write results to')
    parser.add_argument('--log', type=str, default=None, help='Path to log file')
    parser.add_argument('--pdb', action='store_true', help='Run in the debugger')
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.ERROR
    logging.basicConfig(filename=args.log, level=log_level)

    try:
        do_command(args)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        if args.pdb:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            logging.exception("Uncaught exception {}".format(e))
            raise


if __name__ == '__main__':
    main()
