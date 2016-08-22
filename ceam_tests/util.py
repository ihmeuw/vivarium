# ~/ceam/ceam_tests/util.py

from datetime import datetime, timedelta

import pandas as pd
import numpy as np

import pytest

from ceam import config

from ceam.gbd_data.gbd_ms_functions import load_data_from_cache

from ceam.framework.engine import SimulationContext, _step
from ceam.framework.event import Event
from ceam.framework.util import from_yearly, to_yearly

def _cache_loader(func, *args, **kwargs):
    if func.__name__ == 'generate_ceam_population':
        return build_table(0.0, ['age', 'year', 'sex_id', 'rate'])
    else:
        return load_data_from_cache(func, *args, **kwargs)

def setup_simulation(components, population_size = 100):
    simulation = SimulationContext(components)
    simulation.setup()

    start = datetime(1990, 1, 1)
    generate_emitter = simulation.events.get_emitter('generate_population')

    simulation.population.column_lock = False
    generate_emitter(Event(start, range(population_size)))
    simulation.population.column_lock = True

    return simulation

def pump_simulation(simulation, duration=None, iterations=None):
    timestep = timedelta(days=30.5)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time
    iteration_count = 0

    def should_stop():
        if duration is not None:
            if simulation.current_time - start_time >= duration:
                return True
        elif iterations is not None:
            if iteration_count >= iterations:
                return True
        else:
            raise ValueError('Must supply either duration or iterations')

        return False

    while not should_stop():
        iteration_count += 1
        _step(simulation, timestep)



def assert_rate(simulation, expected_rate, value_func, effective_population_func=lambda s:len(s.population.population), dummy_population=None):
    """ Asserts that the rate of change of some property in the simulation matches expectations.

    Parameters
    ----------
    simulation : ceam.engine.Simulation
    value_func
        a function that takes a Simulation and returns the current value of the property to be tested
    effective_population_func
        a function that takes a Simulation and returns the size of the population over which the rate should be measured (ie. living simulants for mortality)
    expected_rate
        The rate of change we expect or a lambda that will take a rate and return a boolean
    population_sample_func
        A function that takes in a population and returns a subset of it which will be used for the test
    """

    timestep = timedelta(days=30)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time

    count = value_func(simulation)
    total_true_rate = 0
    effective_population_size = 0
    for _ in range(10*12):
        effective_population_size += effective_population_func(simulation)
        _step(simulation, timestep)
        new_count = value_func(simulation)
        total_true_rate += new_count - count
        count = new_count

    try:
        assert expected_rate(to_yearly(total_true_rate, timestep*120))
    except TypeError:
        total_expected_rate = from_yearly(expected_rate, timestep)*effective_population_size
        assert abs(total_expected_rate - total_true_rate)/total_expected_rate < 0.1



def build_table(rate, columns=['age', 'year', 'sex', 'rate']):
    rows = []
    start_year = config.getint('simulation_parameters', 'year_start')
    end_year = config.getint('simulation_parameters', 'year_end')
    for age in range(1, 104):
        for year in range(start_year, end_year+1):
            for sex in ['Male', 'Female']:
                rows.append([age, year, sex, rate])
    return pd.DataFrame(rows, columns=columns)


# End.
