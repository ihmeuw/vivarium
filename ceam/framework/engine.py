"""The engine."""
import os
import os.path
import argparse
from time import time
from collections import Iterable
from datetime import datetime, timedelta
from pprint import pformat
import gc
from bdb import BdbQuit

import pandas as pd

from ceam import config

from ceam.analysis import dump_results

from ceam.framework.values import ValuesManager
from ceam.framework.event import EventManager, Event, emits
from ceam.framework.population import PopulationManager, creates_simulants
from ceam.framework.lookup import InterpolatedDataManager
from ceam.framework.components import load, read_component_configuration
from ceam.framework.randomness import RandomnessStream
from ceam.framework.util import collapse_nested_dict

import logging
_log = logging.getLogger(__name__)


class Builder:
    def __init__(self, context):
        self.lookup = context.tables.build_table
        self.value = context.values.get_value
        self.rate = context.values.get_rate
        self.declare_pipeline = context.values.declare_pipeline
        self.modifies_value = context.values.mutator
        self.emitter = context.events.get_emitter
        self.population_view = context.population.get_view
        self.clock = lambda: lambda: context.current_time
        draw_number = config.run_configuration.draw_number
        self.randomness = lambda key: RandomnessStream(key, self.clock(), draw_number)

    def __repr__(self):
        return ("Builder(\nlookup: {},\nvalue: {},\nrate: {},\n".format(self.lookup, self.value, self.rate)
                + "declare_pipeline: {},\nmodifies_value: {},\n".format(self.declare_pipeline, self.modifies_value)
                + "emitter: {},\npopulation_view: {},\n".format(self.emitter, self.population_view)
                + "clock: {},\nrandomness: {}\n)".format(self.clock, self.randomness))


class SimulationContext:
    """context"""
    def __init__(self, components):
        self.components = components
        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = InterpolatedDataManager()
        self.components.extend([self.tables, self.values, self.events, self.population])
        self.current_time = None

    def setup(self):
        builder = Builder(self)
        components = [self.values, self.events, self.population, self.tables] + list(self.components)
        done = set()

        i = 0
        while i < len(components):
            component = components[i]
            if component is None:
                raise ValueError('None in component list. This likely indicates a bug in a factory function')

            if isinstance(component, Iterable):
                # Unpack lists of components so their constituent components get initialized
                components.extend(component)
            if component not in done:
                if hasattr(component, 'configuration_defaults'):
                    # This reapplies configuration from some components but
                    # that shouldn't be a problem.
                    config.read_dict(component.configuration_defaults, layer='component_configs', source=component)
                if hasattr(component, 'setup'):
                    sub_components = component.setup(builder)
                    done.add(component)
                    if sub_components:
                        components.extend(sub_components)
            i += 1
        self.values.setup_components(components)
        self.events.setup_components(components)
        self.population.setup_components(components)

        self.events.get_emitter('post_setup')(None)

    def __repr__(self):
        return ("SimulationContext(\ncomponents: {},\nvalues: {},\n".format(self.components, self.values)
                + "events: {},\npopulation: {},\ntables: {},\n".format(self.events, self.population, self.tables)
                + "current_time: {})".format(self.current_time))


@emits('time_step')
@emits('time_step__prepare')
@emits('time_step__cleanup')
def _step(simulation, time_step, time_step_emitter, time_step__prepare_emitter, time_step__cleanup_emitter):
    _log.debug(simulation.current_time)
    time_step__prepare_emitter(Event(simulation.population.population.index))
    time_step_emitter(Event(simulation.population.population.index))
    time_step__cleanup_emitter(Event(simulation.population.population.index))
    simulation.current_time += time_step


def _get_time(suffix):
    params = config.simulation_parameters
    month, day = 'month' + suffix, 'day' + suffix
    if month in params or day in params:
        if not (month in params and day in params):
            raise ValueError("you must either specify both a month {0} and a day {0} or neither".format(suffix))
        return datetime(params['year_{}'.format(suffix)], params[month], params.day_start)
    else:
        return datetime(params['year_{}'.format(suffix)], 7, 2)


@creates_simulants
@emits('simulation_end')
def event_loop(simulation, simulant_creator, end_emitter):
    start = _get_time('start')
    stop = _get_time('end')
    time_step = config.simulation_parameters.time_step
    time_step = timedelta(days=time_step)

    simulation.current_time = start

    population_size = config.simulation_parameters.population_size

    if config.simulation_parameters.initial_age != '':
        simulant_creator(population_size, population_configuration={
            'initial_age': config.simulation_parameters.initial_age})
    else:
        simulant_creator(population_size)

    while simulation.current_time < stop:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        _step(simulation, time_step)

    end_emitter(Event(simulation.population.population.index))


def setup_simulation(components):
    if not components:
        components = []
    simulation = SimulationContext(load(components + [_step, event_loop]))

    simulation.setup()

    return simulation


def run_simulation(simulation):
    start = time()

    event_loop(simulation)

    metrics = simulation.values.get_value('metrics')
    metrics.source = lambda index: {}
    metrics = metrics(simulation.population.population.index)
    metrics['simulation_run_time'] = time() - start
    return metrics


def configure(draw_number=None, verbose=False, simulation_config=None):
    if simulation_config:
        if isinstance(simulation_config, dict):
            config.read_dict(simulation_config)
        else:
            config.read(simulation_config)

    if draw_number is not None:
        config.run_configuration.set_with_metadata('draw_number', draw_number,
                                                   layer='override', source='command_line_argument')


def run(components):
    config.set_with_metadata('run_configuration.run_id', str(time()), layer='base')
    config.set_with_metadata('run_configuration.run_key', {'draw': config.run_configuration.draw_number}, layer='base')
    simulation = setup_simulation(components)
    metrics = run_simulation(simulation)
    for k, v in collapse_nested_dict(config.run_configuration.run_key.to_dict()):
        metrics[k] = v

    _log.debug(pformat(metrics))

    unused_config_keys = config.unused_keys()
    if unused_config_keys:
        _log.debug("Some configuration keys not used during run: {}".format(unused_config_keys))

    return metrics


def do_command(args):
    if args.command == 'run':
        configure(draw_number=args.draw, verbose=args.verbose, simulation_config=args.config)
        components = read_component_configuration(args.components)
        results = run(components)
        if args.results_path:
            try:
                os.makedirs(os.path.dirname(args.results_path))
            except FileExistsError:
                # Directory already exists, which is fine
                pass
            dump_results(pd.DataFrame([results]), args.results_path)
    elif args.command == 'list_events':
        if args.components:
            component_configurations = read_component_configuration(args.components)
            components = component_configurations['base']['components']
        else:
            components = None
        simulation = setup_simulation(components)
        print(simulation.events.list_events())
    elif args.command == 'print_configuration':
        configure(draw_number=args.draw, verbose=args.verbose, simulation_config=args.config)
        components = read_component_configuration(args.components)
        load(components)
        print(config)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'list_events', 'print_configuration'])
    parser.add_argument('components', nargs='?', default=None, type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None,
                        help='Path to a config file to load which will take precedence over all other configs')
    parser.add_argument('--draw', '-d', type=int, default=0, help='Which GBD draw to use')
    parser.add_argument('--results_path', '-o', type=str, default=None, help='Path to write results to')
    parser.add_argument('--process_number', '-n', type=int, default=1, help='Instance number for this process')
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
