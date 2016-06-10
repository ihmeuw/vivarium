import os.path
from datetime import datetime, timedelta

import pytest

from ceam.engine import Simulation
from ceam.util import from_yearly

def simulation_factory(modules):
    simulation = Simulation()
    for module in modules:
        module.setup()
    simulation.register_modules(modules)
    data_path = os.path.join(str(pytest.config.rootdir), 'ceam_tests', 'test_data')
    simulation.load_population(os.path.join(data_path, 'population_columns'))
    simulation.load_data(data_path)
    simulation._verify_tables(datetime(1990, 1, 1), datetime(2010, 12, 1))
    return simulation

def assert_rate(simulation, count_func, effective_population_func, expected_rate):
    """ Asserts that the rate of change of some property in the simulation matches expectations.

    Parameters
    ----------
    simulation : ceam.engine.Simulation
    count_func
        a function that takes a Simulation and returns the current value of the property to be tested
    effective_population_func
        a function that takes a Simulation and returns the size of the population over which the rate should be measured (ie. living simulants for mortality)
    expected_rate
        The rate of change we expect
    """

    simulation.reset_population()
    timestep = timedelta(days=30)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time

    count = count_func(simulation)
    total_expected_rate = 0
    total_true_rate = 0
    for _ in range(10*12):
        effective_population_size = effective_population_func(simulation)
        simulation._step(timestep)
        new_count = count_func(simulation)
        total_expected_rate += from_yearly(expected_rate, timestep)*effective_population_size
        total_true_rate += new_count - count
        count = new_count

    assert abs(total_expected_rate - total_true_rate)/total_expected_rate < 0.1

