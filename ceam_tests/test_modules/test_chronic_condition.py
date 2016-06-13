import pytest

from ceam.modules.chronic_condition import ChronicConditionModule

from ceam_tests.util import simulation_factory, assert_rate

import numpy as np
np.random.seed(100)

@pytest.mark.data
def test_incidence_rate():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01)])
    assert_rate(simulation, 0.7, lambda sim: (sim.population.no_initial_disease == True).sum(), lambda sim: (sim.population.no_initial_disease == False).sum())

@pytest.mark.data
def test_mortality_rate():
    simulation = simulation_factory([ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)])
    assert_rate(simulation, 0.7, lambda sim: (sim.population.alive == False).sum(), lambda sim: (sim.population.alive == True).sum())

def test_disability_weight():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.0.csv', 0.01)])

    # No one is sick initially so there should be no disability weight
    assert simulation.disability_weight().sum() == 0

    # Get some people sick. Now there should be some weight
    illnesses = [1,2,3,4,5]
    simulation.population.loc[illnesses, ['no_initial_disease']] = True
    assert round(simulation.disability_weight().sum() - 0.01*len(illnesses), 5) == 0
