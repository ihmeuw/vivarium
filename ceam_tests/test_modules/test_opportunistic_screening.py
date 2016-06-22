from datetime import timedelta

import pytest

import pandas as pd

from ceam.events import PopulationEvent

from ceam_tests.util import simulation_factory, pump_simulation, assert_all_equal

from ceam.modules.opportunistic_screening import _hypertensive_categories, OpportunisticScreeningModule, MEDICATIONS

def _population_factory():
    population = []
    population.append((40, 130)) # Normotensive, below 60
    population.append((60, 145)) # Normotensive, exactly 60
    population.append((70, 145)) # Normotensive, above 60

    population.append((40, 140)) # Hypertensive, below 60
    population.append((40, 145)) # Hypertensive, below 60
    population.append((60, 170)) # Hypertensive, exatly 60
    population.append((70, 150)) # Hypertensive, above 60
    population.append((70, 155)) # Hypertensive, above 60

    population.append((40, 185)) # Severe hypertensive, below 60
    population.append((70, 185)) # Severe hypertensive, above 60

    population = pd.DataFrame(population, columns=['age', 'systolic_blood_pressure'])
    population['sex'] = 1
    population['alive'] = True
    population['medication_count'] = 0
    population['healthcare_followup_date'] = None
    population['healthcare_last_visit_date'] = None
    population['fractional_age'] = population['age']
    return population


def test_hypertensive_categories():
    population = _population_factory()

    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(population)

    assert len(normotensive) == 3
    assert len(hypertensive) == 5
    assert len(severe_hypertension) == 2

def test_drug_effects():

def test_drug_cost():
    simulation, module = blood_pressure_setup()

    # No one is taking drugs yet so there should be no cost
    module.emit_event(PopulationEvent('time_step', simulation.population))
    assert module.cost_by_year[simulation.current_time.year] == 0

    # Now everyone is on one drug
    simulation.population.medication_count = 1
    simulation.last_time_step = timedelta(days=30)
    module.emit_event(PopulationEvent('time_step', simulation.population))
    daily_cost_of_first_medication = MEDICATIONS[0]['daily_cost']
    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_first_medication * 30 * len(simulation.population)

    # Now everyone is on all the drugs
    simulation.population.medication_count = len(MEDICATIONS)
    simulation.current_time += timedelta(days=361) # Force us into the next year
    module.emit_event(PopulationEvent('time_step', simulation.population))
    daily_cost_of_all_medication = sum(m['daily_cost'] for m in MEDICATIONS)
    assert round(module.cost_by_year[simulation.current_time.year], 5) == round(daily_cost_of_all_medication * 30 * len(simulation.population), 5)

def test_blood_pressure_test_cost():
    simulation, module = blood_pressure_setup()

    # Everybody goes to the hospital
    simulation.emit_event(PopulationEvent('general_healthcare_access', simulation.population))
    cost_of_a_single_test = simulation.config.getfloat('opportunistic_screening', 'blood_pressure_test_cost')
    assert module.cost_by_year[simulation.current_time.year] == cost_of_a_single_test * len(simulation.population)

    # Later, everybody goes to their followup appointment
    simulation.current_time += timedelta(days=361) # Force us into the next year
    simulation.emit_event(PopulationEvent('followup_healthcare_access', simulation.population))
    cost_of_a_followup = cost_of_a_single_test + simulation.config.getfloat('appointments', 'cost')
    assert module.cost_by_year[simulation.current_time.year] == cost_of_a_followup * len(simulation.population)

@pytest.fixture(scope="module")
def blood_pressure_setup():
    module = OpportunisticScreeningModule()
    simulation = simulation_factory([module])
    dummy_population = _population_factory()

    simulation.deregister_modules([module])
    pump_simulation(simulation, iterations=1, dummy_population=dummy_population)
    simulation.register_modules([module])
    return simulation, module

#NOTE: If these tests start breaking mysteriously, it's likely because something changed the order in which pytest is executing them.
# They must run in the order shown here since they represent a sequence of events with state shared through the blood_pressure_setup fixture.
def test_general_blood_pressure_test(blood_pressure_setup):
    simulation, module = blood_pressure_setup
    event = PopulationEvent('general_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert_all_equal(normotensive.medication_count, 0)
    assert_all_equal(normotensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*60))
    assert_all_equal(hypertensive.medication_count, 0)
    assert_all_equal(hypertensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5))
    assert_all_equal(severe_hypertension.medication_count, 2)
    assert_all_equal(severe_hypertension.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*6))

def test_first_followup_blood_pressure_test(blood_pressure_setup):
    simulation, module = blood_pressure_setup
    simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
    event = PopulationEvent('followup_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert_all_equal(normotensive.medication_count, 0)
    assert_all_equal(normotensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*60))
    assert_all_equal(hypertensive.medication_count, 1)
    assert_all_equal(hypertensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*6))
    assert_all_equal(severe_hypertension.medication_count, 3)
    assert_all_equal(severe_hypertension.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*6))

def test_second_followup_blood_pressure_test(blood_pressure_setup):
    simulation, module = blood_pressure_setup
    simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
    event = PopulationEvent('followup_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert_all_equal(normotensive.medication_count, 0)
    assert_all_equal(normotensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*60))
    assert_all_equal(hypertensive.medication_count, 2)
    assert_all_equal(hypertensive.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*6))
    assert_all_equal(severe_hypertension.medication_count, 4)
    assert_all_equal(severe_hypertension.healthcare_followup_date, simulation.current_time + timedelta(days=30.5*6))

def test_Nth_followup_blood_pressure_test(blood_pressure_setup):
    simulation, module = blood_pressure_setup
    for _ in range(10):
        simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
        event = PopulationEvent('followup_healthcare_access', simulation.population)
        module.emit_event(event)

    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert_all_equal(normotensive.medication_count, 0)
    assert_all_equal(hypertensive.medication_count, len(MEDICATIONS))
    assert_all_equal(severe_hypertension.medication_count, len(MEDICATIONS))
