import argparse
import re
from datetime import datetime, timedelta

import pandas as pd

from ceam import config

from ceam.framework.values import ValuesManager, joint_value_combiner, joint_value_post_processor
from ceam.framework.event import EventManager, Event, emits
from ceam.framework.population import PopulationManager
from ceam.framework.lookup import MergedTableManager
from ceam.framework.components import load, read_component_list

import logging
_log = logging.getLogger(__name__)

class Builder:
    def __init__(self, context):
        self.lookup = context.tables.build_table
        self.value = context.values.get_pipeline
        self.emitter = context.events.get_emitter
        self.population_view = context.population.get_view
        self.clock = lambda: lambda: context.current_time

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
        self.values.declare_pipeline('disability_weight', combiner=joint_value_combiner, post_processing=joint_value_post_processor)
        self.values.declare_pipeline(re.compile('paf\..*'), combiner=joint_value_combiner, post_processing=joint_value_post_processor, source=lambda index: pd.Series(1.0, index=index))
        self.values.declare_pipeline('metrics', source=lambda index: {})
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

@emits('time_step')
@emits('time_step__prepare')
def _step(simulation, time_step, time_step_emitter, time_step__prepare_emitter):
    _log.debug(simulation.current_time)
    time_step__prepare_emitter.emit(Event(simulation.current_time, simulation.population.population.index))
    time_step_emitter.emit(Event(simulation.current_time, simulation.population.population.index))
    simulation.current_time += time_step

@emits('generate_population')
@emits('simulation_end')
def event_loop(simulation, generate_emitter, end_emitter):
    start = config.getint('simulation_parameters', 'year_start')
    start = datetime(start, 1, 1)
    stop = config.getint('simulation_parameters', 'year_end')
    stop = datetime(stop, 12, 30)
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    simulation.population.column_lock = False
    population_size = config.getint('simulation_parameters', 'population_size')
    generate_emitter.emit(Event(start, range(population_size)))
    simulation.population.column_lock = True

    simulation.current_time = start
    while simulation.current_time < stop:
        _step(simulation, time_step)

    end_emitter.emit(Event(simulation.current_time, simulation.population.population.index))

def run_simulation(components):
    component_paths = read_component_list(components)

    simulation = SimulationContext(load(component_paths + [_step, event_loop]))

    simulation.setup()

    event_loop(simulation)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('components', type=str)
    parser.add_argument('--verbose', '-v', action='store_true')
    parser.add_argument('--config', type=str, default=None, help='Path to a config file to load which will take presidence over all other configs')
    parser.add_argument('--draw', '-d', type=int, default=0, help='Which GBD draw to use')
    parser.add_argument('--process_number', '-n', type=int, default=1, help='Instance number for this process')
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        _log.debug('Enabling DEBUG logging')

    if args.config:
        config.read(args.config)

    config.set('run_configuration', 'draw_number', str(args.draw))
    config.set('run_configuration', 'process_number', str(args.process_number))

    run_simulation(args.components)

if __name__ == '__main__':
    main()
