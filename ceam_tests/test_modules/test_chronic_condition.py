import os.path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
np.random.seed(100)

import pytest

from ceam.modules import SimulationModule
from ceam.modules.chronic_condition import ChronicConditionModule, _calculate_time_spent_in_phases

from ceam_tests.util import simulation_factory, assert_rate, pump_simulation

class MetricsModule(SimulationModule):
    def __init__(self, condition):
        super(MetricsModule, self).__init__()
        self.time_to_death = []
        self.condition = condition

    def setup(self):
        self.register_event_listener(self.death_listener, 'deaths')

    def death_listener(self, event):
        # Record time between onset and duration for simulants affected by the condition of interest
        affected_population = event.affected_population[event.affected_population[self.condition]]
        self.time_to_death.extend(self.simulation.current_time.timestamp() - affected_population[self.condition+'_event_time'])

@pytest.mark.parametrize('current_time, onset_time, acute_phase_duration, acute_time, chronic_time', [
    # Acute phase overlaps perfectly with current time step
    [datetime(1990, 12, 1), datetime(1990, 11, 1), timedelta(days=30), timedelta(days=30), timedelta(days=0)],
    # Acute phase overlaps partially with current time step and onset time is equal to the beginning of the current time step
    [datetime(1990, 12, 1), datetime(1990, 11, 1), timedelta(days=28), timedelta(days=28), timedelta(days=2)],
    # Acute phase overlaps partially with current time step and onset time is before the beginning of the current time step
    [datetime(1991, 11, 1), datetime(1991, 9, 15), timedelta(days=28), timedelta(days=11), timedelta(days=19)],
    # Acute phase overlaps partially with current time step and onset time is after the beginning of the current time step
    [datetime(1991, 10, 1), datetime(1991, 9, 15), timedelta(days=28), timedelta(days=14), timedelta(days=0)],
    # Onset was much earlier that current time
    [datetime(1990, 12, 1), datetime(1980, 11, 1), timedelta(days=28), timedelta(days=0), timedelta(days=30)],
    # Onset is in the future
    [datetime(1990, 12, 1), datetime(2021, 11, 1), timedelta(days=28), timedelta(days=0), timedelta(days=0)],
    # Acute phase duration is much longer than time step
    [datetime(1990, 12, 1), datetime(1990, 9, 1), timedelta(days=280), timedelta(days=30), timedelta(days=0)],
    # Acute phase duration is much shorted than time step
    [datetime(1990, 10, 1), datetime(1990, 9, 15), timedelta(days=2), timedelta(days=2), timedelta(days=14)],
    ])
def test_calculate_time_spent_in_phases(acute_phase_duration, onset_time, current_time, acute_time, chronic_time):
    current_time_step = timedelta(days=30)

    onset_times = pd.Series([onset_time.timestamp()])
    times = _calculate_time_spent_in_phases(onset_times, acute_phase_duration, current_time, current_time_step)
    assert timedelta(seconds=int(times.acute.iloc[0])) == acute_time
    assert timedelta(seconds=int(times.chronic.iloc[0])) == chronic_time

@pytest.mark.slow
def test_incidence_rate():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01)])
    assert_rate(simulation, 0.7,
                lambda sim: (sim.population.no_initial_disease).sum(),
                lambda sim: (~sim.population.no_initial_disease).sum()
               )

@pytest.mark.slow
def test_mortality_rate_without_acute_phase():
    simulation = simulation_factory([ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)])
    assert_rate(simulation, 0.7, lambda sim: (~sim.population.alive).sum(), lambda sim: (sim.population.alive).sum())

@pytest.mark.slow
def test_mortality_rate_with_acute_phase_saturated_initial_population():
    # In this test everyone starts with the condition and are out of the acute phase so the only rate should be chronic
    simulation = simulation_factory([
        ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01, acute_mortality_table_name='mortality_1.6.csv')
        ])
    assert_rate(simulation, 0.7, lambda sim: (~sim.population.alive).sum(), lambda sim: (sim.population.alive).sum())

@pytest.mark.slow
def test_that_acute_mortality_shortens_time_between_onset_and_death():
    # First get a baseline with only chronic mortality
    metrics_module = MetricsModule('no_initial_disease')
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01), metrics_module])
    pump_simulation(simulation, duration=timedelta(days=360*5))
    base_time_to_death = np.mean(metrics_module.time_to_death)

    # Now, include acute mortality
    metrics_module = MetricsModule('no_initial_disease')
    simulation = simulation_factory([
        ChronicConditionModule('no_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01, acute_mortality_table_name='mortality_1.6.csv'),
        metrics_module
        ])
    pump_simulation(simulation, duration=timedelta(days=360*5))
    assert np.mean(metrics_module.time_to_death) < base_time_to_death

@pytest.mark.slow
def test_mortality_rate_with_acute_phase_unsaturated_initial_population():
    # In this test people will be having accute events so the rate should be higher
    simulation = simulation_factory([
        ChronicConditionModule('no_initial_disease', 'mortality_0.7.csv', 'incidence_0.7.csv', 0.01, acute_mortality_table_name='mortality_1.6.csv')
        ])
    assert_rate(simulation,
                lambda rate: rate > 0.7,
                lambda sim: (~sim.population.alive).sum(),
                lambda sim: (sim.population.alive).sum()
               )

def test_disability_weight():
    simulation = simulation_factory([ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.0.csv', 0.01)])

    # No one is sick initially so there should be no disability weight
    assert simulation.disability_weight().sum() == 0

    # Get some people sick. Now there should be some weight
    illnesses = [1, 2, 3, 4, 5]
    simulation.population.loc[illnesses, ['no_initial_disease']] = True
    assert round(simulation.disability_weight().sum() - 0.01*len(illnesses), 5) == 0

def test_initial_column_data():
    data_path = os.path.join(str(pytest.config.rootdir), 'ceam_tests', 'test_data', 'population_columns')

    module = ChronicConditionModule('high_initial_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)
    module.load_population_columns(data_path, 1000)

    # There is a initial column file with this name. Make sure we got data
    assert module.population_columns.high_initial_disease.sum() > 0

    module = ChronicConditionModule('some_crazy_disease', 'mortality_0.7.csv', 'incidence_0.0.csv', 0.01)
    module.load_population_columns(data_path, 1000)
    # There is no initial column file with this name. Make sure we didn't get data
    assert module.population_columns.some_crazy_disease.sum() == 0

    module = ChronicConditionModule('some_crazy_disease',
                                    'mortality_0.7.csv',
                                    'incidence_0.0.csv',
                                    0.01,
                                    initial_column_table_name='high_initial_disease.csv'
                                   )
    module.load_population_columns(data_path, 1000)
    # Even though there's no initial data for this condition we've specified an alternate table to load so we should use the data from that
    assert module.population_columns.some_crazy_disease.sum() > 0

    module = ChronicConditionModule('some_crazy_disease',
                                    'mortality_0.7.csv',
                                    'incidence_0.0.csv',
                                    0.01,
                                    initial_column_table_name='some_crazy_disease.csv'
                                   )
    with pytest.raises(OSError):
        # We're explicitly specifying a table to load but it doesn't exist. Fail fast
        module.load_population_columns(data_path, 1000)

@pytest.mark.slow
def test_multiple_events():
    simulation = simulation_factory([
        ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01, allow_multiple_events=True)
        ])
    assert_rate(simulation, 0.7, lambda sim: sim.population.no_initial_disease_event_count.sum())
    assert simulation.population.no_initial_disease_event_count.mean() > 1

@pytest.mark.slow
def test_suppresion_of_multiple_events():
    simulation = simulation_factory([
        ChronicConditionModule('no_initial_disease', 'mortality_0.0.csv', 'incidence_0.7.csv', 0.01, allow_multiple_events=False)
        ])
    assert np.all(simulation.population.no_initial_disease_event_count <= 1)
