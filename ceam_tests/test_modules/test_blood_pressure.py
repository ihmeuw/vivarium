import pytest
from datetime import timedelta

from ceam_tests.util import simulation_factory, pump_simulation

from ceam.modules.blood_pressure import BloodPressureModule

import numpy as np
np.random.seed(100)

@pytest.mark.data
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

    # Check that blood pressure goes up over time in our cohort
    assert simulation.population.systolic_blood_pressure.mean() > initial_mean_sbp
    # And that there's still no one wildly out of bounds
    assert ((simulation.population.systolic_blood_pressure > (sbp_mean+interval)) | ( simulation.population.systolic_blood_pressure < (sbp_mean-interval))).sum() == 0
