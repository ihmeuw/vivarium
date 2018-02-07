"""The engine."""
import argparse
from bdb import BdbQuit
import gc
import os
import os.path
from pprint import pformat, pprint
from time import time
from typing import Mapping
from collections import namedtuple

import yaml
import pandas as pd

from vivarium.config_tree import ConfigTree
from vivarium.framework.values import ValuesManager
from vivarium.framework.event import EventManager, Event, emits
from vivarium.framework.population import PopulationManager, creates_simulants
from vivarium.framework.lookup import InterpolatedDataManager
from vivarium.framework.components import load_component_manager
from vivarium.framework.randomness import RandomnessManager
from vivarium.framework.util import collapse_nested_dict
from vivarium.framework.results_writer import get_results_writer

import logging
_log = logging.getLogger(__name__)


class SimulationContext:
    """context"""
    def __init__(self, component_manager, configuration):
        self.component_manager = component_manager
        self.configuration = configuration
        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = InterpolatedDataManager()
        self.randomness = RandomnessManager()
        self.current_time = None
        self.step_size = pd.Timedelta(0, unit='D')

    def update_time(self):
        """Updates the simulation clock."""
        self.current_time += self.step_size

    def setup(self):
        builder = Builder(self)

        self.component_manager.add_components(
            [self.values, self.events, self.population, self.tables, self.randomness])
        self.component_manager.load_components_from_config()
        self.component_manager.setup_components(builder)

        self.values.setup_components(self.component_manager.components)
        self.events.setup_components(self.component_manager.components)
        self.population.setup_components(self.component_manager.components)

        self.events.get_emitter('post_setup')(None)

    def __repr__(self):
        return "SimulationContext(current_time={}, step_size={})".format(self.current_time, self.step_size)


class Builder:
    """Useful tools for constructing and configuring simulation components."""
    def __init__(self, context: SimulationContext):
        self.lookup = context.tables.build_table
        self.value = context.values.get_value
        self.rate = context.values.get_rate
        self.modifies_value = context.values.mutator
        self.emitter = context.events.get_emitter
        self.population_view = context.population.get_view
        self.clock = lambda: lambda: context.current_time
        self.step_size = lambda: lambda: context.step_size
        self.configuration = context.configuration
        self.randomness = namedtuple(
            'Randomness', ['get_stream', 'register_simulants'])(context.randomness.get_randomness_stream,
                                                                context.randomness.register_simulants)


    def __repr__(self):
        return "Builder()"


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


def _get_time(suffix, params):
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
    sim_params = simulation.configuration.simulation_parameters
    pop_params = simulation.configuration.population

    step_size = sim_params.time_step
    simulation.step_size = pd.Timedelta(days=step_size//1, hours=(step_size % 1)*24)
    start = _get_time('start', sim_params)
    stop = _get_time('end', sim_params)

    # Fencepost the creation of the initial population.
    simulation.current_time = start - simulation.step_size
    population_size = pop_params.population_size
    simulant_creator(population_size)
    simulation.update_time()

    while simulation.current_time < stop:
        gc.collect()  # TODO: Actually figure out where the memory leak is.
        _step(simulation)

    end_emitter(Event(simulation.population.population.index))


def setup_simulation(component_manager, config):
    config.run_configuration.set_with_metadata('run_id', str(time()), layer='base')
    config.run_configuration.set_with_metadata('run_key',
                                               {'draw': config.run_configuration.input_draw_number}, layer='base')
    component_manager.add_components([_step, event_loop])
    simulation = SimulationContext(component_manager, config)
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


def build_base_configuration(parameters: Mapping = None) -> ConfigTree:
    config = ConfigTree(layers=['base', 'component_configs', 'model_override', 'override'])
    if os.path.exists(os.path.expanduser('~/vivarium.yaml')):
        config.load(os.path.expanduser('~/vivarium.yaml'), layer='override',
                    source=os.path.expanduser('~/vivarium.yaml'))

    default_metadata = {'layer': 'base', 'source': os.path.realpath(__file__)}

    # Some setup for the defaults
    def _get_draw_template(draw_type_, value_):
        return {'run_configuration': {f'{draw_type_}_number': value_}}

    # Get an input and model draw
    for draw_type in ['input_draw', 'model_draw']:
        if parameters and draw_type in parameters and parameters[draw_type] is not None:
            metadata = {'layer': 'override', 'source': 'command line or launching script'}
            draw = _get_draw_template(draw_type, parameters[draw_type])
        else:
            metadata = default_metadata
            draw = _get_draw_template(draw_type, 0)
        config.update(draw, **metadata)

    # FIXME: Hack in some stuff from the config in ceam-inputs until we can clean this up. -J.C.
    config.update(
        {
            'simulation_parameters':
                {
                    'year_start': 2005,
                    'year_end': 2010,
                    'time_step': 1
                },
            'input_data':
                {
                    'location_id': 180
                },

        }, **default_metadata)

    if parameters and 'results_path' in parameters:
        config.update({'run_configuration': {'results_directory': parameters['results_path']}})
    if config.run_configuration.results_directory is None:
        config.run_configuration.results_directory = os.path.expanduser('~/vivarium_results/')

    return config


def build_simulation_configuration(parameters: Mapping) -> ConfigTree:
    """Builds a configuration from the on disk user configuration, command line arguments,
    and component configurations passed in by file path.

    Parameters
    ----------
    parameters :
        Dictionary possibly containing keys:
            'input_draw': Input draw number to use,
            'model_draw': Model draw number to use,
            'components': Component configuration (file path, yaml string, or dict),
            'config': Configuration overrides (file path, yaml string, or dict)

    Returns
    -------
    A valid simulation configuration.
    """
    # Start with the base configuration in the user's home directory
    config = build_base_configuration(parameters)

    default_component_manager = {'vivarium': {'component_manager': 'vivarium.framework.components.ComponentManager'}}
    default_dataset_manager = {'vivarium': {'dataset_manager': 'vivarium.framework.components.DummyDatasetManager'}}
    default_metadata = {'layer': 'base', 'source': os.path.realpath(__file__)}

    # Set any configuration overrides from component and branch configurations.
    config.update(parameters.get('config', None), layer='override')  # source is implicit
    config.update(parameters.get('components', None), layer='model_override')  # source is implicit
    if 'configuration' in config:
        config.configuration.source = parameters.get('components', None)

    # Make sure we have a component and dataset manager
    if 'component_manager' not in config['vivarium']:
        config.update(default_component_manager, **default_metadata)
    if 'dataset_manager' not in config['vivarium']:
        config.update(default_dataset_manager, **default_metadata)

    return config


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
        results_writer = get_results_writer(config.run_configuration.results_directory, args.components)
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
