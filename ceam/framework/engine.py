import os
import os.path
import argparse
from time import time
import re
from datetime import datetime, timedelta
from pprint import pformat

import numpy as np
import pandas as pd

from ceam import config

from ceam.analysis import analyze_results, dump_results

from ceam.framework.values import ValuesManager, set_combiner, joint_value_combiner, joint_value_post_processor, rescale_post_processor, NullValue
from ceam.framework.event import EventManager, Event, emits
from ceam.framework.population import PopulationManager
from ceam.framework.lookup import MergedTableManager
from ceam.framework.components import load, read_component_configuration
from ceam.framework.randomness import RandomnessStream

import logging
_log = logging.getLogger(__name__)

class Builder:
    def __init__(self, context):
        self.lookup = context.tables.build_table
        self.value = context.values.get_pipeline
        self.modifies_value = context.values.mutator
        self.emitter = context.events.get_emitter
        self.population_view = context.population.get_view
        self.clock = lambda: lambda: context.current_time
        draw_number = config.getint('run_configuration', 'draw_number')
        self.randomness = lambda key: RandomnessStream(key, self.clock(), draw_number)

class SimulationContext:
    def __init__(self, components):
        self.components = components
        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = MergedTableManager()
        self.components.extend([self.tables, self.values, self.events, self.population])
        self.current_time = None

    def setup(self):
        self.values.declare_pipeline('disability_weight',
                combiner=joint_value_combiner,
                post_processor=lambda a: rescale_post_processor(joint_value_post_processor(a)),
                source=lambda index: pd.Series(1.0, index=index))

        self.values.declare_pipeline(re.compile('paf\..*'),
                combiner=joint_value_combiner,
                post_processor=joint_value_post_processor,
                source=lambda index: NullValue(index))

        self.values.declare_pipeline(re.compile('modelable_entity_ids\..*'),
                combiner=set_combiner,
                post_processor=None,
                source=lambda: set())

        self.values.declare_pipeline('metrics', post_processor=None, source=lambda index: {})
        builder = Builder(self)
        components = list(self.components)
        i = 0
        while i < len(components):
            component = components[i]
            if hasattr(component, 'setup'):
                sub_components = component.setup(builder)
                if sub_components:
                    components.extend(sub_components)
            i += 1
        self.values.setup_components(components)
        self.events.setup_components(components)
        self.population.setup_components(components)
        self.tables.setup_components(components)

        self.events.get_emitter('post_setup')(None)

@emits('time_step')
@emits('time_step__prepare')
@emits('time_step__cleanup')
def _step(simulation, time_step, time_step_emitter, time_step__prepare_emitter, time_step__cleanup_emitter):
    _log.debug(simulation.current_time)
    time_step__prepare_emitter.emit(Event(simulation.current_time, simulation.population.population.index))
    time_step_emitter.emit(Event(simulation.current_time, simulation.population.population.index))
    time_step__cleanup_emitter.emit(Event(simulation.current_time, simulation.population.population.index))
    simulation.current_time += time_step

@emits('generate_population')
@emits('post_setup')
@emits('simulation_end')
def event_loop(simulation, generate_emitter, post_setup_emitter, end_emitter):
    start = config.getint('simulation_parameters', 'year_start')
    start = datetime(start, 1, 1)
    stop = config.getint('simulation_parameters', 'year_end')
    stop = datetime(stop, 12, 30)
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    population_size = config.getint('simulation_parameters', 'population_size')
    generate_emitter.emit(Event(start, range(population_size)))
    simulation.population.initialized = True

    simulation.current_time = start
    while simulation.current_time < stop:
        _step(simulation, time_step)

    end_emitter.emit(Event(simulation.current_time, simulation.population.population.index))

def setup_simulation(components):
    if not components:
        components = []
    simulation = SimulationContext(load(components + [_step, event_loop]))
    simulation.setup()
    return simulation

def run_simulation(simulation):
    start = time()

    event_loop(simulation)

    metrics = simulation.values.get_pipeline('metrics')(simulation.population.population.index)
    metrics['duration'] = time() - start
    return metrics

def configure(draw_number=0, verbose=False, simulation_config=None):
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        _log.debug('Enabling DEBUG logging')

    if simulation_config:
        if isinstance(config, dict):
            config.read_dict(config)
        else:
            config.read(simulation_config)

    config.set('run_configuration', 'draw_number', str(draw_number))

def run_comparison(component_config, results_path=None):
    component_configurations = read_component_configuration(component_config)
    all_metrics = []
    for configuration in component_configurations.values():
        _log.debug('Starting comparison: {}'.format(configuration['name']))
        simulation = setup_simulation(configuration['components'])
        metrics = run_simulation(simulation)
        metrics['comparison'] = configuration['name']
        _log.debug(pformat(metrics))
        all_metrics.append(metrics)
    if results_path:
        try:
            os.makedirs(os.path.dirname(results_path))
        except FileExistsError:
            # Directory already exists, which is fine
            pass
        dump_results(pd.DataFrame([all_metrics]), results_path)

def run_configuration(component_config, results_path=None, sub_configuration_name='base'):
    component_configurations = read_component_configuration(component_config)
    configuration = component_configurations[sub_configuration_name]
    simulation = setup_simulation(configuration['components'])
    metrics = run_simulation(simulation)
    metrics['comparison'] = configuration['name']
    if results_path:
        try:
            os.makedirs(os.path.dirname(results_path))
        except FileExistsError:
            # Directory already exists, which is fine
            pass
        dump_results(pd.DataFrame([metrics]), results_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['run', 'list_events'])
    parser.add_argument('components', nargs='?', default=None, type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--config', '-c', type=str, default=None, help='Path to a config file to load which will take presidence over all other configs')
    parser.add_argument('--draw', '-d', type=int, default=0, help='Which GBD draw to use')
    parser.add_argument('--process_number', '-n', type=int, default=1, help='Instance number for this process')
    parser.add_argument('--results_path', '-o', type=str, default=None, help='File to write output to')
    args = parser.parse_args()

    if args.command == 'run':
        configure(draw_number=args.draw, verbose=args.verbose, simulation_config=args.config)
        run_comparison(args.components, results_path=args.results_path)
    elif args.command == 'list_events':
        if args.components:
            component_configurations = read_component_configuration(args.components)
            components = component_configurations['base']['components']
        else:
            components = None
        simulation = setup_simulation(components)
        print(simulation.events.list_events())

if __name__ == '__main__':
    main()
