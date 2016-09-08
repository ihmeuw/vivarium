# ~/ceam/ceam_tests/test_modules/test_healthcare_access_module.py

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ceam import config
from ceam_tests.util import setup_simulation, assert_rate, build_table

from ceam.framework.event import listens_for

from ceam.components.healthcare_access import HealthcareAccess
from ceam.components.base_population import generate_base_population, adherence

import numpy as np

np.random.seed(100)


class Metrics:
    def setup(self, builder):
        self.access_count = 0

    @listens_for('general_healthcare_access')
    def count_access(self, event):
        self.access_count += len(event.index)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
@patch('ceam.components.healthcare_access.load_data_from_cache')
def test_general_access(utilization_rate_mock):
    utilization_rate_mock.side_effect = lambda *args, **kwargs: build_table(0.1, ['age', 'year', 'sex', 'utilization_proportion'])
    metrics = Metrics()
    simulation = setup_simulation([generate_base_population, adherence, metrics, HealthcareAccess()])

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)


#TODO: get fixture data for the cost table so we can test in a stable space
#@pytest.mark.slow
#def test_general_access_cost():
#    metrics = MetricsModule()
#    access = HealthcareAccessModule()
#    simulation = simulation_factory([metrics, access])
#
#    simulation.reset_population()
#    timestep = timedelta(days=30)
#    start_time = datetime(1990, 1, 1)
#    simulation.current_time = start_time
#
#    simulation._step(timestep)
#    simulation._step(timestep)
#    simulation._step(timestep)
#
#    assert np.allclose(sum(access.cost_by_year.values()) / metrics.access_count, access.appointment_cost[1990])


# End.
