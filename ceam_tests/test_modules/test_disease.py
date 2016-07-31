# ~/ceam/ceam_tests/test_modules/test_disease.py

import pytest
from unittest.mock import Mock, patch

import pandas as pd
import numpy as np

from ceam_tests.util import simulation_factory, pump_simulation, build_table

from ceam.util import from_yearly

from ceam.state_machine import Transition, State
from ceam.modules.disease import DiseaseState, IncidenceRateTransition, ExcessMortalityState, DiseaseModule


def test_dwell_time():
    test_state = DiseaseState('test', 0.0, dwell_time=10)
    test_state.parent = Mock()
    test_state.parent.parent = None
    test_state.parent.current_time.timestamp.return_value = 0

    done_state = State('done')
    test_state.transition_set.append(Transition(done_state))

    population = pd.DataFrame({'state': ['test']*10, 'test_event_time': [0]*10, 'test_event_count': [0]*10, 'simulant_id':range(10)})

    population = test_state.next_state(population, 'state')
    assert np.all(population.state == 'test')

    test_state.root.current_time.timestamp.return_value = 20

    population = test_state.next_state(population, 'state')
    assert np.all(population.state == 'done')
    assert np.all(population.test_event_time == 20)
    assert np.all(population.test_event_count == 1)

@patch('ceam.modules.disease.get_excess_mortality')
def test_mortality_rate(mock_get_excess_mortality):
    mock_get_excess_mortality.side_effect = lambda meid: build_table(0.7)
    module = DiseaseModule('test_disease')
    healthy = State('healthy')
    mortality_state = ExcessMortalityState('sick', modelable_entity_id=0, disability_weight=0.1)

    healthy.transition_set.append(Transition(mortality_state))

    module.states.extend([healthy, mortality_state])

    simulation = simulation_factory([module])

    # Initial mortality should just be the base rates because everybody starts healthy
    initial_mortality = simulation.mortality_rates(simulation.population)

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.all(((initial_mortality + from_yearly(0.7, simulation.last_time_step)).round(4) ==  simulation.mortality_rates(simulation.population).round(4)))


@patch('ceam.modules.disease.get_incidence')
def test_incidence(mock_get_incidence):
    mock_get_incidence.side_effect = lambda meid: build_table(0.7)
    module = DiseaseModule('test_disease')
    healthy = State('healthy')
    sick = State('sick')

    healthy.transition_set.append(IncidenceRateTransition(sick, 'test_incidence', modelable_entity_id=0))

    module.states.extend([healthy, sick])

    simulation = simulation_factory([module])

    assert np.all(from_yearly(0.7, simulation.last_time_step) == simulation.incidence_rates(simulation.population, 'test_incidence'))


def test_load_population_default_columns():
    module = DiseaseModule('test_disease')
    dwell_test = DiseaseState('dwell_test', disability_weight=0.0, dwell_time=10)

    module.states.append(dwell_test)

    simulation = simulation_factory([module])

    assert 'dwell_test_event_time' in simulation.population
    assert 'dwell_test_event_count' in simulation.population
    assert np.all(simulation.population.dwell_test_event_count == 0)
    assert np.all(simulation.population.dwell_test_event_time == 0)


def test_load_population_custom_columns():
    module = DiseaseModule('test_disease')
    dwell_test = DiseaseState('dwell_test', disability_weight=0.0, dwell_time=10, event_time_column='special_test_time', event_count_column='special_test_count')

    module.states.append(dwell_test)

    simulation = simulation_factory([module])

    assert 'special_test_time' in simulation.population
    assert 'special_test_count' in simulation.population
    assert np.all(simulation.population.special_test_count == 0)
    assert np.all(simulation.population.special_test_time == 0)


# End.
