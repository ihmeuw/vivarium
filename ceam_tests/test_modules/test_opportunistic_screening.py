from datetime import timedelta

import pytest

import pandas as pd

from ceam import config
from ceam.events import PopulationEvent

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.opportunistic_screening import _hypertensive_categories, OpportunisticScreeningModule, MEDICATIONS
from ceam.modules.healthcare_access import HealthcareAccessModule
from ceam.modules.blood_pressure import BloodPressureModule

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
    simulation, module = screening_setup()

    starting_sbp = simulation.population.systolic_blood_pressure

    # No one is taking any drugs yet so there should be no effect on SBP
    module.adjust_blood_pressure(PopulationEvent('time_step', simulation.population))
    assert (starting_sbp == simulation.population.systolic_blood_pressure).all()

    # Now everyone is on the first drug
    simulation.population['medication_count'] = 1
    module.adjust_blood_pressure(PopulationEvent('time_step', simulation.population))
    assert (starting_sbp > simulation.population.systolic_blood_pressure).all()
    efficacy = MEDICATIONS[0]['efficacy'] * config.getfloat('opportunistic_screening', 'adherence')
    assert (starting_sbp == (simulation.population.systolic_blood_pressure + efficacy)).all()

    # Now everyone is on the first three drugs
    simulation.population['medication_count'] = 3
    simulation.population.systolic_blood_pressure = starting_sbp
    module.adjust_blood_pressure(PopulationEvent('time_step', simulation.population))
    efficacy = sum(m['efficacy'] * config.getfloat('opportunistic_screening', 'adherence') for m in MEDICATIONS[:3])
    assert (starting_sbp == (simulation.population.systolic_blood_pressure + efficacy)).all()

def test_dependencies():
    # NOTE: This is kind of a silly test since it just verifies that the class is defined correctly but I did actually make that mistake. -Alec
    simulation, module = screening_setup()
    ordered_module_ids = [m.module_id() for m in simulation._ordered_modules]
    assert BloodPressureModule in ordered_module_ids
    assert HealthcareAccessModule in ordered_module_ids

def test_drug_cost():
    simulation, module = screening_setup()

    # No one is taking drugs yet so there should be no cost
    module.emit_event(PopulationEvent('time_step', simulation.population))
    assert module.cost_by_year[simulation.current_time.year] == 0

    # Now everyone is on one drug
    simulation.population['medication_count'] = 1
    simulation.last_time_step = timedelta(days=30)
    module.emit_event(PopulationEvent('time_step', simulation.population))
    daily_cost_of_first_medication = MEDICATIONS[0]['daily_cost']
    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_first_medication * 30 * len(simulation.population)

    # Now everyone is on all the drugs
    simulation.population['medication_count'] = len(MEDICATIONS)
    simulation.current_time += timedelta(days=361) # Force us into the next year
    module.emit_event(PopulationEvent('time_step', simulation.population))
    daily_cost_of_all_medication = sum(m['daily_cost'] for m in MEDICATIONS)
    assert round(module.cost_by_year[simulation.current_time.year], 5) == round(daily_cost_of_all_medication * 30 * len(simulation.population), 5)

def test_blood_pressure_test_cost():
    simulation, module = screening_setup()

    # Everybody goes to the hospital
    simulation.emit_event(PopulationEvent('general_healthcare_access', simulation.population))
    cost_of_a_single_test = config.getfloat('opportunistic_screening', 'blood_pressure_test_cost')
    assert module.cost_by_year[simulation.current_time.year] == cost_of_a_single_test * len(simulation.population)

    # Later, everybody goes to their followup appointment
    simulation.current_time += timedelta(days=361) # Force us into the next year
    simulation.emit_event(PopulationEvent('followup_healthcare_access', simulation.population))
    cost_of_a_followup = cost_of_a_single_test + config.getfloat('appointments', 'cost')
    assert module.cost_by_year[simulation.current_time.year] == cost_of_a_followup * len(simulation.population)

@pytest.fixture(scope="module")
def screening_setup():
    module = OpportunisticScreeningModule()
    simulation = simulation_factory([module])
    dummy_population = _population_factory()

    simulation.remove_children([module])
    pump_simulation(simulation, iterations=1, dummy_population=dummy_population)
    simulation.add_children([module])
    return simulation, module

#NOTE: If these tests start breaking mysteriously, it's likely because something changed the order in which pytest is executing them.
# They must run in the order shown here since they represent a sequence of events with state shared through the screening_setup fixture.
def test_general_blood_pressure_test(screening_setup):
    simulation, module = screening_setup
    event = PopulationEvent('general_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert (normotensive.medication_count == 0).all()
    assert (normotensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*60)).all()
    assert (hypertensive.medication_count == 0).all()
    assert (hypertensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5)).all()
    assert (severe_hypertension.medication_count == 2).all()
    assert (severe_hypertension.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*6)).all()

def test_first_followup_blood_pressure_test(screening_setup):
    simulation, module = screening_setup
    simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
    event = PopulationEvent('followup_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert (normotensive.medication_count == 0).all()
    assert (normotensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*60)).all()
    assert (hypertensive.medication_count == 1).all()
    assert (hypertensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*6)).all()
    assert (severe_hypertension.medication_count == 3).all()
    assert (severe_hypertension.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*6)).all()

def test_second_followup_blood_pressure_test(screening_setup):
    simulation, module = screening_setup
    simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
    event = PopulationEvent('followup_healthcare_access', simulation.population)
    module.emit_event(event)
    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert (normotensive.medication_count == 0).all()
    assert (normotensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*60)).all()
    assert (hypertensive.medication_count == 2).all()
    assert (hypertensive.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*6)).all()
    assert (severe_hypertension.medication_count == 4).all()
    assert (severe_hypertension.healthcare_followup_date == simulation.current_time + timedelta(days=30.5*6)).all()

def test_Nth_followup_blood_pressure_test(screening_setup):
    simulation, module = screening_setup
    for _ in range(10):
        simulation.current_time += timedelta(days=30) # Tick forward without triggering any actual events
        event = PopulationEvent('followup_healthcare_access', simulation.population)
        module.emit_event(event)

    normotensive, hypertensive, severe_hypertension = _hypertensive_categories(simulation.population)
    assert (normotensive.medication_count == 0).all()
    assert (hypertensive.medication_count == len(MEDICATIONS)).all()
    assert (severe_hypertension.medication_count == len(MEDICATIONS)).all()
