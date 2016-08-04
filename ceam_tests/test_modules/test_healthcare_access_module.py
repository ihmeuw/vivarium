# ~/ceam/ceam_tests/test_modules/test_healthcare_access_module.py

from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from ceam import config
from ceam_tests.util import simulation_factory, assert_rate, build_table
from ceam.engine import SimulationModule

from ceam.modules.healthcare_access import HealthcareAccessModule

import numpy as np

np.random.seed(100)


class MetricsModule(SimulationModule):
    def setup(self):
        self.register_event_listener(self.count_access, 'general_healthcare_access')
        self.access_count = 0

    def count_access(self, event):
        self.access_count += len(event.affected_population)

    def reset(self):
        self.access_count = 0


@pytest.mark.slow
@patch('ceam.modules.healthcare_access.load_data_from_cache')
def test_general_access(utilization_rate_mock):
    utilization_rate_mock.side_effect = lambda *args, **kwargs: build_table(0.1, ['age', 'year', 'sex', 'utilization_proportion'])
    metrics = MetricsModule()
    simulation = simulation_factory([metrics, HealthcareAccessModule()])
    initial_population = simulation.population

    # 1.2608717447575932 == a monthly probability 0.1 as a yearly rate
    assert_rate(simulation, 1.2608717447575932, lambda s: metrics.access_count)


@pytest.mark.slow
def test_general_access_cost():
    metrics = MetricsModule()
    access = HealthcareAccessModule()
    simulation = simulation_factory([metrics, access])

    simulation.reset_population()
    timestep = timedelta(days=30)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time

    simulation._step(timestep)
    simulation._step(timestep)
    simulation._step(timestep)

    assert np.allclose(sum(access.cost_by_year.values()) / metrics.access_count, access.appointment_cost[1990])
    


# End.
