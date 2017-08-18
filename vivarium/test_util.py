import numbers

import pandas as pd
import numpy as np

from vivarium import config

from vivarium.framework.engine import SimulationContext, _step
from vivarium.framework.event import listens_for
from vivarium.framework.population import uses_columns
from vivarium.framework.util import from_yearly, to_yearly
from vivarium.framework import randomness


def setup_simulation(components, population_size=100, start=None):
    simulation = SimulationContext(components)
    simulation.setup()
    if start:
        simulation.current_time = start
    else:
        year_start = config.simulation_parameters.year_start
        simulation.current_time = pd.Timestamp(year_start, 1, 1)

    if 'initial_age' in config.simulation_parameters:
        simulation.population._create_simulants(population_size,
                                                population_configuration={
                                                    'initial_age': config.simulation_parameters.initial_age})
    else:
        simulation.population._create_simulants(population_size)

    return simulation


def pump_simulation(simulation, time_step_days=None, duration=None, iterations=None):
    if duration is None and iterations is None:
        raise ValueError('Must supply either duration or iterations')

    if time_step_days:
        config.simulation_parameters.time_step = time_step_days
    simulation.step_size = pd.Timedelta(config.simulation_parameters.time_step, unit='D')

    if duration is not None:
        if isinstance(duration, numbers.Number):
            duration = pd.Timedelta(days=duration)
        time_step = pd.Timedelta(days=config.simulation_parameters.time_step)
        iterations = int(duration / time_step) + 1

    if run_from_ipython():
        for _ in log_progress(range(iterations), name='Step'):
            _step(simulation)
    else:
        for i in range(iterations):
            _step(simulation)

    return iterations

def run_from_ipython():
    """Taken from https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python"""
    try:
        __IPYTHON__
        return True
    except NameError:
        return  False

def assert_rate(simulation, expected_rate, value_func,
                effective_population_func=lambda s: len(s.population.population), dummy_population=None):
    """ Asserts that the rate of change of some property in the simulation matches expectations.

    Parameters
    ----------
    simulation : vivarium.engine.Simulation
    value_func
        a function that takes a Simulation and returns the current value of the property to be tested
    effective_population_func
        a function that takes a Simulation and returns the size of the population over which the rate should be measured (ie. living simulants for mortality)
    expected_rate
        The rate of change we expect or a lambda that will take a rate and return a boolean
    population_sample_func
        A function that takes in a population and returns a subset of it which will be used for the test
    """
    start_time = pd.Timestamp(1990, 1, 1)
    time_step = pd.Timedelta(30, unit='D')
    simulation.current_time = start_time
    simulation.step_size = time_step

    count = value_func(simulation)
    total_true_rate = 0
    effective_population_size = 0
    for _ in range(10*12):
        effective_population_size += effective_population_func(simulation)
        _step(simulation)
        new_count = value_func(simulation)
        total_true_rate += new_count - count
        count = new_count

    try:
        assert expected_rate(to_yearly(total_true_rate, time_step*120))
    except TypeError:
        total_expected_rate = from_yearly(expected_rate, time_step)*effective_population_size
        assert abs(total_expected_rate - total_true_rate)/total_expected_rate < 0.1


def build_table(value, columns=('age', 'year', 'sex', 'rate')):
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
    return pd.DataFrame(rows, columns=['age', 'year', 'sex'] + list(value_columns))


@listens_for('initialize_simulants', priority=0)
@uses_columns(['age', 'sex', 'location', 'alive', 'entrance_time', 'exit_time'])
def generate_test_population(event):
    initial_age = event.user_data.get('initial_age', None)
    population = pd.DataFrame(index=event.index)

    if 'pop_age_start' in config.simulation_parameters:
        age_start = config.simulation_parameters.pop_age_start
    else:
        age_start = 0

    if 'pop_age_end' in config.simulation_parameters:
        age_end = config.simulation_parameters.pop_age_end
    else:
        age_end = 100

    if initial_age is not None and initial_age != '':
        population['age'] = initial_age
        population['age'] = population['age'].astype(float)
    else:
        population['age'] = (randomness.random('test_population_age', population.index)
                             * (age_end - age_start) + age_start)

    population['sex'] = randomness.choice('test_population_sex'+str(config.run_configuration.draw_number),
                                          population.index, ['Male', 'Female'])
    population['alive'] = 'alive'
    if 'location_id' in config.simulation_parameters:
        population['location'] = config.simulation_parameters.location_id
    else:
        population['location'] = 180

    population['entrance_time'] = event.time
    population['exit_time'] = pd.NaT

    event.population_view.update(population)


@listens_for('time_step')
@uses_columns(['age'], "alive == 'alive'")
def age_simulants(event):
    event.population['age'] += event.step_size.days / 365.0
    event.population_view.update(event.population)


def make_dummy_column(name, initial_value):
    @listens_for('initialize_simulants')
    @uses_columns([name])
    def make_column(event):
        event.population_view.update(pd.Series(initial_value, index=event.index, name=name))
    return make_column


def get_randomness(key='test', clock=lambda: pd.Timestamp(1990, 7, 2), seed=12345):
    return randomness.RandomnessStream(key, clock, seed=seed)


def log_progress(sequence, every=None, size=None, name='Items'):
    """Taken from https://github.com/alexanderkuk/log-progress"""
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
