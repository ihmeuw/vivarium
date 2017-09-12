"""The engine."""
import os
import os.path
import argparse
from time import time
from collections import Iterable
from pprint import pformat
import gc
from bdb import BdbQuit

import pandas as pd

from vivarium import config

from vivarium.framework.values import ValuesManager
from vivarium.framework.event import EventManager, Event, emits
from vivarium.framework.population import PopulationManager, creates_simulants
from vivarium.framework.lookup import InterpolatedDataManager
from vivarium.framework.components import load, read_component_configuration
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.util import collapse_nested_dict
from vivarium.framework.results_writer import ResultsWriter

import logging
_log = logging.getLogger(__name__)


class Builder:
    """Useful tools for constructing and configuring simulation components."""
    def __init__(self, context):
        self.lookup = context.tables.build_table
        self.value = context.values.get_value
        self.rate = context.values.get_rate
        self.modifies_value = context.values.mutator
        self.emitter = context.events.get_emitter
        self.population_view = context.population.get_view
        self.clock = lambda: lambda: context.current_time
        self.step_size = lambda: lambda: context.step_size
        input_draw_number = config.run_configuration.draw_number
        model_draw_number = config.run_configuration.model_draw_number
        self.randomness = lambda key: RandomnessStream(key, self.clock(), (input_draw_number, model_draw_number))

    def __repr__(self):
        return "Builder()"


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
        self.step_size = pd.Timedelta(0, unit='D')

    def update_time(self):
        """Updates the simulation clock."""
        self.current_time += self.step_size

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
        return "SimulationContext(components={}, current_time={}, step_size={})".format(self.components,
                                                                                        self.current_time,
                                                                                        self.step_size)


@emits('time_step')
@emits('time_step__prepare')
@emits('time_step__cleanup')
@emits('collect_metrics')
def _step(simulation, time_step_emitter, time_step__prepare_emitter,
          time_step__cleanup_emitter, collect_metrics_emitter):
    _log.debug(simulation.current_time)
    time_step__prepare_emitter(Event(simulation.population.population.index))
    time_step_emitter(Event(simulation.population.population.index))
    time_step__cleanup_emitter(Event(simulation.population.population.index))
    collect_metrics_emitter(Event(simulation.population.population.index))
    simulation.update_time()


def _get_time(suffix):
    params = config.simulation_parameters
    month, day = 'month' + suffix, 'day' + suffix
    if month in params or day in params:
        if not (month in params and day in params):
            raise ValueError("you must either specify both a month {0} and a day {0} or neither".format(suffix))
        return pd.Timestamp(params['year_{}'.format(suffix)], params[month], params.day_start)
    else:
        return pd.Timestamp(params['year_{}'.format(suffix)], 7, 2)


@creates_simulants
@emits('simulation_end')
def event_loop(simulation, simulant_creator, end_emitter):
    start = _get_time('start')
    stop = _get_time('end')

    simulation.current_time = start

    population_size = config.simulation_parameters.population_size

    if config.simulation_parameters.initial_age is not None and config.simulation_parameters.pop_age_start is None:
        simulant_creator(population_size, population_configuration={
            'initial_age': config.simulation_parameters.initial_age})
    else:
        simulant_creator(population_size)

    simulation.step_size = pd.Timedelta(config.simulation_parameters.time_step, unit='D')

    while simulation.current_time < stop:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        _step(simulation)

    end_emitter(Event(simulation.population.population.index))


def setup_simulation(components):
    config.run_configuration.set_with_metadata('run_id', str(time()), layer='base')
    config.run_configuration.set_with_metadata('run_key',
                                               {'draw': config.run_configuration.draw_number}, layer='base')
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


def configure(input_draw_number=None, model_draw_number=None, verbose=False, simulation_config=None):
    if simulation_config:
        if isinstance(simulation_config, dict):
            config.read_dict(simulation_config)
        else:
            config.read(simulation_config)

    if input_draw_number is not None:
        config.run_configuration.set_with_metadata('draw_number', input_draw_number,
                                                   layer='override', source='command_line_argument')
    else:
        if 'draw_number' not in config.run_configuration:
            config.run_configuration.set_with_metadata('draw_number', 0,
                                                       layer='override', source='default')

    if model_draw_number is not None:
        config.run_configuration.set_with_metadata('model_draw_number', model_draw_number,
                                                   layer='override', source='command_line_argument')
    else:
        if 'model_draw_number' not in config.run_configuration:
            config.run_configuration.set_with_metadata('model_draw_number', 0,
                                                       layer='override', source='default')


def run(simulation):
    metrics = run_simulation(simulation)
    for k, v in collapse_nested_dict(config.run_configuration.run_key.to_dict()):
        metrics[k] = v

    _log.debug(pformat(metrics))

    unused_config_keys = config.unused_keys()
    if unused_config_keys:
        _log.debug("Some configuration keys not used during run: %s", unused_config_keys)

    return metrics, simulation.population._population


def do_command(args):
    if args.command == 'run':
        configure(input_draw_number=args.input_draw, verbose=args.verbose, simulation_config=args.config)
        components = read_component_configuration(args.components)
        simulation = setup_simulation(components)
        results, final_state = run(simulation)
        if args.results_path:
            results_root = os.path.dirname(args.results_path)
            rw = ResultsWriter(results_root)
            rw.dump_simulation_configuration(args.components)
            rw.write_output(pd.DataFrame([results]), 'output.hdf')
            rw.write_output(final_state, 'final_state.hdf')
    elif args.command == 'list_events':
        if args.components:
            component_configurations = read_component_configuration(args.components)
            components = component_configurations['base']['components']
        else:
            components = None
        simulation = setup_simulation(components)
        print(simulation.events.list_events())
    elif args.command == 'print_configuration':
        configure(input_draw_number=args.input_draw, verbose=args.verbose, simulation_config=args.config)
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
    parser.add_argument('--input_draw', '-d', type=int, default=0, help='Which GBD draw to use')
    parser.add_argument('--model_draw', type=int, default=0, help="Which draw from the model's own variation to use")
    parser.add_argument('--results_path', '-o', type=str, default=None, help='Output directory to write results to')
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
