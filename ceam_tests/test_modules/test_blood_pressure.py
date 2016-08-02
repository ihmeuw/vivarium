# ~/ceam/ceam_tests/test_modules/test_blood_pressure.py

import pytest
from datetime import timedelta

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.disease_models import heart_disease_factory, stroke_factory
import numpy as np

np.random.seed(100)


@pytest.mark.slow
def test_basic_SBP_bounds():
    simulation = simulation_factory([BloodPressureModule()])

    sbp_mean = 138 # Mean across all demographics
    sbp_std = 15 # Standard deviation across all demographics
    interval = sbp_std * 3.5
    pump_simulation(simulation, iterations=1) # Get blood pressure stablaized

    #Check that no one is wildly out of range
    assert ((simulation.population.systolic_blood_pressure > (sbp_mean+interval)) | ( simulation.population.systolic_blood_pressure < (sbp_mean-interval))).sum() == 0

    initial_mean_sbp = simulation.population.systolic_blood_pressure.mean()

    pump_simulation(simulation, duration=timedelta(days=5*365))

    # Check that blood pressure goes up over time as our cohort ages
    assert simulation.population.systolic_blood_pressure.mean() > initial_mean_sbp
    # And that there's still no one wildly out of bounds
    assert ((simulation.population.systolic_blood_pressure > (sbp_mean+interval)) | ( simulation.population.systolic_blood_pressure < (sbp_mean-interval))).sum() == 0


@pytest.mark.parametrize('condition_module, rate_label', [(heart_disease_factory(), 'heart_attack'), (stroke_factory(), 'hemorrhagic_stroke'), (stroke_factory(), 'ischemic_stroke'), ])
@pytest.mark.slow
def test_blood_pressure_effect_on_incidince(condition_module, rate_label):
    bp_module = BloodPressureModule()
    simulation = simulation_factory([bp_module, condition_module])

    pump_simulation(simulation, iterations=1) # Get blood pressure stablaized
    simulation.remove_children([bp_module])

    # Base incidence rate without blood pressure
    base_incidence = simulation.incidence_rates(simulation.population, rate_label)

    simulation.add_children([bp_module])

    # Get incidence including the effect of blood pressure
    bp_incidence = simulation.incidence_rates(simulation.population, rate_label)

    # Blood pressure should only increase rates
    assert base_incidence.mean() < bp_incidence.mean()

    pump_simulation(simulation, duration=timedelta(days=5*365))

    # Increase in incidence should rise over time as the cohort ages and SBP increases
    assert bp_incidence.mean() < simulation.incidence_rates(simulation.population, rate_label).mean()


# End.
