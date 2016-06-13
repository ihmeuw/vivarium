from datetime import datetime, timedelta

import pytest

from ceam_tests.util import simulation_factory, assert_rate
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

@pytest.mark.data
def test_general_access():
    metrics = MetricsModule()
    simulation = simulation_factory([metrics, HealthcareAccessModule()])
    initial_population = simulation.population

    # Men and women have different utilization rates
    simulation.population = initial_population[initial_population.sex == 1]
    assert_rate(simulation, simulation.config.getfloat('appointments', 'male_utilization_rate'), lambda s: metrics.access_count)
    simulation.population = initial_population[initial_population.sex == 2]
    assert_rate(simulation, simulation.config.getfloat('appointments', 'male_utilization_rate'), lambda s: metrics.access_count)

@pytest.mark.data
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

    assert round(sum(access.cost_by_year.values()) / metrics.access_count, 5) == simulation.config.getfloat('appointments', 'cost')
