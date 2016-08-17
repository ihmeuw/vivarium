import argparse
from datetime import datetime, timedelta

from ceam import config

from ceam.framework.values import ValuesManager
from ceam.framework.event import EventManager, Event, emits
from ceam.framework.population import PopulationManager
from ceam.framework.lookup import MergedTableManager
from ceam.framework.components import load, read_component_list

class Builder:
    def __init__(self, lookup, value, emitter):
        self.lookup = lookup
        self.value = value
        self.emitter = emitter

class SimulationContext:
    def __init__(self, components):
        self.components = components
        self.values = ValuesManager()
        self.events = EventManager()
        self.population = PopulationManager()
        self.tables = MergedTableManager()
        self.components.extend([self.tables, self.values, self.events, self.population])
        self.current_time = None
        self.time_step = None

    def setup(self):
        builder = Builder(self.tables.build_table, self.values.get_pipeline, self.events.get_emitter)
        for component in self.components:
            if hasattr(component, 'setup'):
                component.setup(builder)
        self.values.setup_components(self.components)
        self.events.setup_components(self.components)
        self.population.setup_components(self.components)
        self.tables.setup_components(self.components)

@emits('time_step')
@emits('time_step__prepare')
def _step(simulation, time_step_emitter, time_step__prepare_emitter):
    time_step__prepare_emitter.emit(Event(simulation.current_time, simulation.time_step, simulation.population.population.index))
    time_step_emitter.emit(Event(simulation.current_time, simulation.time_step, simulation.population.population.index))
    simulation.current_time += time_step

@emits('generate_population')
@emits('simulation_end')
def event_loop(simulation, start, stop, time_step, generate_emitter, end_emitter):
    simulation.population.column_lock = False
    population_size = config.getint('simulation_parameters', 'population_size')
    generate_emitter.emit(Event(start, time_step, range(population_size)))
    simulation.population.column_lock = True

    simulation.current_time = start
    simulation.time_step = time_step
    while simulation.current_time < stop:
        _step(simulation)

    end_emitter.emit(Event(simulation.current_time, simulation.time_step, simulation.population.population.index))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('components', type=str)
    args = parser.parse_args()

    start = datetime(1990, 1, 1)
    stop = datetime(2010, 12, 1)
    time_step = timedelta(days=30.5)

    component_paths = read_component_list(args.components)

    simulation = SimulationContext(load(component_paths + [_step, event_loop]))

    simulation.setup()

    event_loop(simulation, start, stop, time_step)
