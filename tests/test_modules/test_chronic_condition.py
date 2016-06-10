import os.path
from datetime import datetime, timedelta

import pytest

from ceam.engine import Simulation
from ceam.modules.chronic_condition import ChronicConditionModule
from ceam.util import from_yearly

def simulation_factory(modules):
    simulation = Simulation()
    for module in modules:
        module.setup()
    simulation.register_modules(modules)
    data_path = os.path.join(str(pytest.config.rootdir), 'tests', 'test_data')
    simulation.load_population(os.path.join(data_path, 'population_columns'))
    simulation.load_data(data_path)
    simulation._verify_tables(datetime(1990, 1, 1), datetime(1995, 12, 1))
    return simulation

@pytest.mark.data
def test_incidence_rate():
    simulation = simulation_factory([ChronicConditionModule('test_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01)])
    simulation.reset_population()
    timestep = timedelta(days=30)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time

    disease_count = (simulation.population.test_disease == True).sum()
    expected_rate = 0
    true_rate = 0
    for _ in range(10*12):
        vulnerable_population = (simulation.population.test_disease == False).sum()
        simulation._step(timestep)
        new_disease_count = (simulation.population.test_disease == True).sum()
        expected_rate += from_yearly(0.7, timestep)*vulnerable_population
        true_rate += new_disease_count - disease_count
        disease_count = new_disease_count


    assert abs(expected_rate - true_rate)/expected_rate < 0.1
