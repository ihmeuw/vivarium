# ~/ceam/ceam_tests/test_modules/test_disease.py

import pytest
from unittest.mock import Mock, patch
from datetime import timedelta

import pandas as pd
import numpy as np

from ceam import config

from ceam_tests.util import setup_simulation, pump_simulation, build_table

from ceam.framework.util import from_yearly

from ceam.components.base_population import generate_base_population

from ceam.framework.state_machine import Transition, State
from ceam.framework.event import Event
from ceam.framework.disease import DiseaseState, IncidenceRateTransition, ExcessMortalityState, DiseaseModel


@patch('ceam.framework.disease.get_disease_states')
def test_dwell_time(get_disease_states_mock):
    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('state')
    healthy_state = State('healthy')
    event_state = DiseaseState('event', 0.0, dwell_time=timedelta(days=28))
    healthy_state.transition_set.append(Transition(event_state))

    done_state = State('sick')
    event_state.transition_set.append(Transition(done_state))
    model.states.extend([healthy_state, event_state, done_state])

    simulation = setup_simulation([generate_base_population, model], population_size=10)
    emitter = simulation.events.get_emitter('time_step')

    # Move everyone into the event state
    emitter(Event(simulation.current_time, simulation.population.population.index))
    event_time = simulation.current_time

    assert np.all(simulation.population.population.state == 'event')

    simulation.current_time += timedelta(days=10)

    # Not enough time has passed for people to move out of the event state, so they should all still be there
    emitter(Event(simulation.current_time, simulation.population.population.index))

    assert np.all(simulation.population.population.state == 'event')

    simulation.current_time += timedelta(days=20)

    # Now enough time has passed so people should transition away
    emitter(Event(simulation.current_time, simulation.population.population.index))

    assert np.all(simulation.population.population.state == 'sick')

    assert np.all(simulation.population.population.event_event_time == event_time.timestamp())
    assert np.all(simulation.population.population.event_event_count == 1)


def test_mortality_rate():
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    mortality_state = ExcessMortalityState('sick', modelable_entity_id=2412, disability_weight=0.1)

    healthy.transition_set.append(Transition(mortality_state))

    model.states.extend([healthy, mortality_state])

    simulation = setup_simulation([generate_base_population, model])

    mortality_state.mortality = simulation.tables.build_table(build_table(0.7))

    mortality_rate = simulation.values.get_rate('mortality_rate')
    mortality_rate.source = simulation.tables.build_table(build_table(0.0))

    pump_simulation(simulation, iterations=1)

    # Folks instantly transition to sick so now our mortality rate should be much higher
    assert np.allclose(from_yearly(0.7, time_step), mortality_rate(simulation.population.population.index))



@patch('ceam.framework.disease.get_disease_states')
def test_incidence(get_disease_states_mock):
    time_step = config.getfloat('simulation_parameters', 'time_step')
    time_step = timedelta(days=time_step)

    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    healthy = State('healthy')
    sick = State('sick')

    transition = IncidenceRateTransition(sick, 'test_incidence', modelable_entity_id=2412)
    healthy.transition_set.append(transition)

    model.states.extend([healthy, sick])

    simulation = setup_simulation([generate_base_population, model])

    transition.base_incidence = simulation.tables.build_table(build_table(0.7))

    incidence_rate = simulation.values.get_rate('incidence_rate.test_incidence')

    pump_simulation(simulation, iterations=1)

    assert np.all(from_yearly(0.7, time_step) == incidence_rate(simulation.population.population.index))

@patch('ceam.framework.disease.get_disease_states')
def test_load_population_custom_columns(get_disease_states_mock):
    get_disease_states_mock.side_effect = lambda population, state_map: pd.DataFrame({'condition_state': 'healthy'}, index=population.index)
    model = DiseaseModel('test_disease')
    dwell_test = DiseaseState('dwell_test', disability_weight=0.0, dwell_time=10, event_time_column='special_test_time', event_count_column='special_test_count')

    model.states.append(dwell_test)

    simulation = setup_simulation([generate_base_population, model])

    assert 'special_test_time' in simulation.population.population
    assert 'special_test_count' in simulation.population.population
    assert np.all(simulation.population.population.special_test_count == 0)
    assert np.all(simulation.population.population.special_test_time == 0)


# End.
