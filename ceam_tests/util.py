from datetime import datetime, timedelta

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.engine import SimulationContext, _step
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns
from ceam.framework.util import from_yearly, to_yearly
from ceam.framework import randomness


def setup_simulation(components, population_size=100, start=None):
    simulation = SimulationContext(components)
    simulation.setup()
    if start:
        simulation.current_time = start
    else:
        year_start = config.simulation_parameters.year_start
        simulation.current_time = datetime(year_start, 1, 1)

    if 'initial_age' in config.simulation_parameters:
        simulation.population._create_simulants(population_size,
                                                population_configuration={
                                                    'initial_age': config.simulation_parameters.initial_age})
    else:
        simulation.population._create_simulants(population_size)

    return simulation


def pump_simulation(simulation, time_step_days=None, duration=None, iterations=None):
    if time_step_days:
        config.simulation_parameters.time_step = time_step_days
    time_step = timedelta(days=float(config.simulation_parameters.time_step))
    start_time = simulation.current_time
    iteration_count = 0

    def should_stop():
        if duration is not None:
            if simulation.current_time - start_time >= duration:
                return True
        elif iterations is not None:
            if iteration_count >= iterations:
                return True
        else:
            raise ValueError('Must supply either duration or iterations')

        return False

    while not should_stop():
        iteration_count += 1
        _step(simulation, time_step)

    return iteration_count


def assert_rate(simulation, expected_rate, value_func, effective_population_func=lambda s:len(s.population.population), dummy_population=None):
    """ Asserts that the rate of change of some property in the simulation matches expectations.

    Parameters
    ----------
    simulation : ceam.engine.Simulation
    value_func
        a function that takes a Simulation and returns the current value of the property to be tested
    effective_population_func
        a function that takes a Simulation and returns the size of the population over which the rate should be measured (ie. living simulants for mortality)
    expected_rate
        The rate of change we expect or a lambda that will take a rate and return a boolean
    population_sample_func
        A function that takes in a population and returns a subset of it which will be used for the test
    """

    timestep = timedelta(days=30)
    start_time = datetime(1990, 1, 1)
    simulation.current_time = start_time

    count = value_func(simulation)
    total_true_rate = 0
    effective_population_size = 0
    for _ in range(10*12):
        effective_population_size += effective_population_func(simulation)
        _step(simulation, timestep)
        new_count = value_func(simulation)
        total_true_rate += new_count - count
        count = new_count

    try:
        assert expected_rate(to_yearly(total_true_rate, timestep*120))
    except TypeError:
        total_expected_rate = from_yearly(expected_rate, timestep)*effective_population_size
        assert abs(total_expected_rate - total_true_rate)/total_expected_rate < 0.1


def build_table(value, columns=['age', 'year', 'sex', 'rate']):
    value_columns = columns[3:]
    if not isinstance(value, list):
        value = [value]*len(value_columns)

    if len(value) != len(value_columns):
        raise ValueError('Number of values must match number of value columns')

    rows = []
    start_year = config.simulation_parameters.year_start
    end_year = config.simulation_parameters.year_end
    for age in range(0, 140):
        for year in range(start_year, end_year+1):
            for sex in ['Male', 'Female']:
                r_values = []
                for v in value:
                    if v is None:
                        r_values.append(np.random.random())
                    elif callable(v):
                        r_values.append(v(age, sex, year))
                    else:
                        r_values.append(v)
                rows.append([age, year, sex] + r_values)
    return pd.DataFrame(rows, columns=columns)


@listens_for('initialize_simulants', priority=0)
@uses_columns(['age', 'fractional_age', 'sex', 'location', 'alive'])
def generate_test_population(event):
    population_size = len(event.index)
    initial_age = event.user_data.get('initial_age', None)
    population = pd.DataFrame(index=range(population_size))

    if 'pop_age_start' in config.simulation_parameters:
        age_start = config.simulation_parameters.pop_age_start
    else:
        age_start = 0

    if 'pop_age_end' in config.simulation_parameters:
        age_end = config.simulation_parameters.pop_age_end
    else:
        age_end = 100

    if initial_age is not None and initial_age is not '':
        population['fractional_age'] = initial_age
    else:
        population['fractional_age'] = randomness.random('test_population_age'+str(config.run_configuration.draw_number), population.index) * (age_end - age_start) + age_start
    population['fractional_age'] = population['fractional_age'].astype(float)
    population['age'] = population['fractional_age'].astype(int)

    population['sex'] = randomness.choice('test_population_sex'+str(config.run_configuration.draw_number), population.index, ['Male', 'Female'])
    population['alive'] = True
    if 'location_id' in config.simulation_parameters:
        population['location'] = config.simulation_parameters.location_id
    else:
        population['location'] = 180

    event.population_view.update(population)

