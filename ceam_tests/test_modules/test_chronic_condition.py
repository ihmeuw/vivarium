import os.path

import pytest

from ceam.modules.chronic_condition import ChronicConditionModule

from ceam_tests.util import simulation_factory, assert_rate

import numpy as np
np.random.seed(100)

@pytest.mark.slow
def test_incidence_rate():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01)])
    assert_rate(simulation, 0.7, lambda sim: (sim.population.no_initial_disease == True).sum(), lambda sim: (sim.population.no_initial_disease == False).sum())

@pytest.mark.slow
def test_mortality_rate_without_acute_phase():
    simulation = simulation_factory([ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)])
    assert_rate(simulation, 0.7, lambda sim: (sim.population.alive == False).sum(), lambda sim: (sim.population.alive == True).sum())

@pytest.mark.slow
def test_mortality_rate_with_acute_phase_saturated_initial_population():
    # In this test everyone starts with the condition and are out of the acute phase so the only rate should be chronic
    simulation = simulation_factory([ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01, acute_mortality_table_name='mortality_1.6.csv')])
    assert_rate(simulation, 0.7, lambda sim: (sim.population.alive == False).sum(), lambda sim: (sim.population.alive == True).sum())

@pytest.mark.slow
def test_mortality_rate_with_acute_phase_unsaturated_initial_population():
    # In this test people will be having accute events so the rate should be higher
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01, acute_mortality_table_name='mortality_1.6.csv')])
    assert_rate(simulation, lambda rate: rate > 0.7, lambda sim: (sim.population.alive == False).sum(), lambda sim: (sim.population.alive == True).sum())

def test_disability_weight():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.0.csv', 0.01)])

    # No one is sick initially so there should be no disability weight
    assert simulation.disability_weight().sum() == 0

    # Get some people sick. Now there should be some weight
    illnesses = [1,2,3,4,5]
    simulation.population.loc[illnesses, ['no_initial_disease']] = True
    assert round(simulation.disability_weight().sum() - 0.01*len(illnesses), 5) == 0

def test_initial_column_data():
    data_path = os.path.join(str(pytest.config.rootdir), 'ceam_tests', 'test_data', 'population_columns')

    module = ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)
    module.load_population_columns(data_path, 1000)

    # There is a initial column file with this name. Make sure we got data
    assert module.population_columns.high_initial_disease.sum() > 0

    module = ChronicConditionModule('some_crazy_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)
    module.load_population_columns(data_path, 1000)
    # There is no initial column file with this name. Make sure we didn't get data
    assert module.population_columns.some_crazy_disease.sum() == 0

    module = ChronicConditionModule('some_crazy_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01, initial_column_table_name='high_initial_disease.csv')
    module.load_population_columns(data_path, 1000)
    # Even though there's no initial data for this condition we've specified an alternate table to load so we should use the data from that
    assert module.population_columns.some_crazy_disease.sum() > 0

    module = ChronicConditionModule('some_crazy_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01, initial_column_table_name='some_crazy_disease.csv')
    with pytest.raises(OSError):
        # We're explicitly specifying a table to load but it doesn't exist. Fail fast
        module.load_population_columns(data_path, 1000)
