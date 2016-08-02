# ~/ceam/ceam_tests/test_modules/test_opportunistic_screening.py

from datetime import timedelta

import pytest

import numpy as np
import pandas as pd

from ceam import config
from ceam.events import PopulationEvent

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.opportunistic_screening import _hypertensive_categories, OpportunisticScreeningModule, MEDICATIONS
from ceam.modules.healthcare_access import HealthcareAccessModule
from ceam.modules.blood_pressure import BloodPressureModule


def _population_setup(population=None):
    if population is None:
        population = pd.DataFrame(index=range(10))
    else:
        population = population.ix[population.index[:10]]

    age_sbps = []
    age_sbps.append((40, 130)) # Normotensive, below 60
    age_sbps.append((60, 145)) # Normotensive, exactly 60
    age_sbps.append((70, 145)) # Normotensive, above 60

    age_sbps.append((40, 140)) # Hypertensive, below 60
    age_sbps.append((40, 145)) # Hypertensive, below 60
    age_sbps.append((60, 170)) # Hypertensive, exactly 60
    age_sbps.append((70, 150)) # Hypertensive, above 60
    age_sbps.append((70, 155)) # Hypertensive, above 60

    age_sbps.append((40, 185)) # Severe hypertensive, below 60
    age_sbps.append((70, 185)) # Severe hypertensive, above 60

    ages, sbps = zip(*age_sbps)
    population['age'] = ages
    population['systolic_blood_pressure'] = sbps

    population['fractional_age'] = population['age']
    return population


def test_hypertensive_categories():
    population = _population_setup()

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
    for medication in MEDICATIONS:
        simulation.population[medication['name']+'_supplied_until'] = simulation.current_time
    module.adjust_blood_pressure(PopulationEvent('time_step', simulation.population))
    assert (starting_sbp[simulation.population.adherence_category == 'adherent'] > simulation.population.systolic_blood_pressure[simulation.population.adherence_category == 'adherent']).all()
    efficacy = MEDICATIONS[0]['efficacy']
    adherent_population = simulation.population[simulation.population.adherence_category == 'adherent']
    assert (starting_sbp[adherent_population.index] == (adherent_population.systolic_blood_pressure + efficacy)).all()

    # Now everyone is on the first three drugs
    simulation.population['medication_count'] = 3
    simulation.population.systolic_blood_pressure = starting_sbp
    module.adjust_blood_pressure(PopulationEvent('time_step', simulation.population))
    efficacy = sum(m['efficacy'] for m in MEDICATIONS[:3])
    adherent_population = simulation.population[simulation.population.adherence_category == 'adherent']
    assert (starting_sbp[adherent_population.index].round() == (adherent_population.systolic_blood_pressure + efficacy).round()).all()


def test_dependencies():
    # NOTE: This is kind of a silly test since it just verifies that the class is defined correctly but I did actually make that mistake. -Alec
    simulation, module = screening_setup()
    ordered_module_ids = [m.module_id() for m in simulation._ordered_modules]
    assert str(BloodPressureModule) in ordered_module_ids
    assert str(HealthcareAccessModule) in ordered_module_ids


def test_medication_cost():
    simulation, module = screening_setup()

    # No one is taking drugs yet so there should be no cost
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=60)
    module._medication_costs(simulation.population)
    assert module.cost_by_year[simulation.current_time.year] == 0
    for medication in MEDICATIONS:
        assert np.all(simulation.population[medication['name']+'_supplied_until'].isnull())

    # Now everyone is on one drug
    simulation.population['medication_count'] = 1
    simulation.last_time_step = timedelta(days=30)
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=60)
    module._medication_costs(simulation.population)

    daily_cost_of_first_medication = MEDICATIONS[0]['daily_cost']
    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_first_medication * 60 * len(simulation.population)
    for medication in MEDICATIONS[1:]:
        assert np.all(simulation.population[medication['name'] + '_supplied_until'].isnull())
    assert np.all(simulation.population[MEDICATIONS[0]['name'] + '_supplied_until'] == simulation.current_time + timedelta(days=60))

    # Now everyone is on all the drugs
    simulation.population['medication_count'] = len(MEDICATIONS)
    simulation.current_time += timedelta(days=361) # Force us into the next year
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=60)
    module._medication_costs(simulation.population)
    daily_cost_of_all_medication = sum(m['daily_cost'] for m in MEDICATIONS)
    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_all_medication * 60 * len(simulation.population)
    for medication in MEDICATIONS[1:]:
        assert np.all(simulation.population[medication['name'] + '_supplied_until'] == simulation.current_time + timedelta(days=60))

    #Now everyone comes back early so they don't need a full sized refill
    simulation.current_time += timedelta(days=45)
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=60)
    module.cost_by_year[simulation.current_time.year] = 0
    module._medication_costs(simulation.population)

    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_all_medication * 45 * len(simulation.population)
    for medication in MEDICATIONS[1:]:
        assert np.all(simulation.population[medication['name'] + '_supplied_until'] == simulation.current_time + timedelta(days=60))

    # This time people come back early again and this time they get a shorter follow up than before.
    simulation.current_time += timedelta(days=1)
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=10)
    module.cost_by_year[simulation.current_time.year] = 0
    module._medication_costs(simulation.population)

    # Cost should be zero because they have plenty of medication left
    assert module.cost_by_year[simulation.current_time.year] == 0
    for medication in MEDICATIONS[1:]:
        assert np.all(simulation.population[medication['name'] + '_supplied_until'] == simulation.current_time + timedelta(days=59))

    # Now they come back for their next appointment and they should have some drugs left over
    simulation.current_time += timedelta(days=10)
    simulation.population['healthcare_followup_date'] = simulation.current_time + timedelta(days=50)
    module.cost_by_year[simulation.current_time.year] = 0
    module._medication_costs(simulation.population)

    assert module.cost_by_year[simulation.current_time.year] == daily_cost_of_all_medication * 1 * len(simulation.population)
    for medication in MEDICATIONS[1:]:
        assert np.all(simulation.population[medication['name'] + '_supplied_until'] == simulation.current_time + timedelta(days=50))


def test_blood_pressure_test_cost():
    simulation, module = screening_setup()

    # For the sake of this test, everyone is healthy so we don't have to worry about them getting prescribed drugs
    # which will change our costs.
    simulation.population['systolic_blood_pressure'] = 112

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
    dummy_population = _population_setup(simulation.population)

    simulation.remove_children([module])
    pump_simulation(simulation, iterations=1, dummy_population=dummy_population)
    simulation.add_children([module])
    return simulation, module


# NOTE: If these tests start breaking mysteriously, it's likely because something changed the order in which pytest is executing them.
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


# End.
