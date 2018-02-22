import numbers

import pandas as pd
import numpy as np

from vivarium.framework.engine import SimulationContext, build_simulation_configuration
from vivarium.framework.event import listens_for, Event
from vivarium.framework.util import from_yearly, to_yearly
from vivarium.framework import randomness
from vivarium.framework.components import load_component_manager


def setup_simulation(components, population_size=100, start=None, input_config=None):
    config = build_simulation_configuration({}) if not input_config else input_config
    component_manager = load_component_manager(config)
    component_manager.add_components(components)
    simulation = SimulationContext(component_manager, config)
    simulation.setup()

    step_size = config.simulation_parameters.time_step
    simulation.step_size = pd.Timedelta(days=step_size // 1, hours=(step_size % 1) * 24)
    if not start:
        start = pd.Timestamp(config.simulation_parameters.year_start, 1, 1)

    # Fencepost the creation of the initial population.
    simulation.current_time = start - simulation.step_size
    simulation.population._create_simulants(population_size)
    simulation.update_time()

    return simulation


def pump_simulation(simulation, time_step_days=None, duration=None, iterations=None, with_logging=True):
    if duration is None and iterations is None:
        raise ValueError('Must supply either duration or iterations')

    if time_step_days:
        simulation.configuration.simulation_parameters.time_step = time_step_days
    simulation.step_size = pd.Timedelta(simulation.configuration.simulation_parameters.time_step, unit='D')

    if duration is not None:
        if isinstance(duration, numbers.Number):
            duration = pd.Timedelta(days=duration)
        time_step = pd.Timedelta(days=simulation.configuration.simulation_parameters.time_step)
        iterations = int(np.ceil(duration / time_step))

    if run_from_ipython() and with_logging:
        for _ in log_progress(range(iterations), name='Step'):
            simulation.step()
    else:
        for i in range(iterations):
            simulation.step()
    end_emitter = simulation.events.get_emitter('simulation_end')
    end_emitter(Event(simulation.population.population.index))

    return iterations


def run_from_ipython():
    """Taken from https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python"""
    try:
        __IPYTHON__
        return True
    except NameError:
        return False


def assert_rate(simulation, expected_rate, value_func,
                effective_population_func=lambda s: len(s.population.population)):
    """ Asserts that the rate of change of some property in the simulation matches expectations.

    Parameters
    ----------
    simulation : vivarium.engine.Simulation
    value_func
        a function that takes a Simulation and returns the current value of the property to be tested
    effective_population_func
        a function that takes a Simulation and returns the size of the population
        over which the rate should be measured (ie. living simulants for mortality)
    expected_rate
        The rate of change we expect or a lambda that will take a rate and return a boolean
    """
    start_time = pd.Timestamp(simulation.configuration.simulation_parameters.year_start, 1, 1)
    time_step = pd.Timedelta(30, unit='D')
    simulation.configuration.simulation_parameters.time_step = time_step
    simulation.current_time = start_time
    simulation.step_size = time_step

    count = value_func(simulation)
    total_true_rate = 0
    effective_population_size = 0
    for _ in range(10*12):
        effective_population_size += effective_population_func(simulation)
        simulation.step()
        new_count = value_func(simulation)
        total_true_rate += new_count - count
        count = new_count

    try:
        assert expected_rate(to_yearly(total_true_rate, time_step*120))
    except TypeError:
        total_expected_rate = from_yearly(expected_rate, time_step)*effective_population_size
        assert abs(total_expected_rate - total_true_rate)/total_expected_rate < 0.1


def build_table(value, year_start, year_end, columns=('age', 'year', 'sex', 'rate')):
    value_columns = columns[3:]
    if not isinstance(value, list):
        value = [value]*len(value_columns)

    if len(value) != len(value_columns):
        raise ValueError('Number of values must match number of value columns')

    rows = []
    for age in range(0, 140):
        for year in range(year_start, year_end+1):
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


class NonCRNTestPopulation:

    configuration_defaults = {
        'population': {
            'age_start': 0,
            'age_end': 100,
        },
        'input_data': {
            'location_id': 180,
        }
    }

    def setup(self, builder):
        self.config = builder.configuration
        self.randomness = builder.randomness.get_stream('population_age_fuzz')
        self.population_view = builder.population.get_view(
            ['age', 'sex', 'location', 'alive', 'entrance_time', 'exit_time'])

    @listens_for('initialize_simulants', priority=0)
    def generate_test_population(self, event):
        age_start = event.user_data.get('age_start', self.config.population.age_start)
        age_end = event.user_data.get('age_end', self.config.population.age_end)
        location = self.config.input_data.location_id

        population = _non_crn_build_population(event.index, age_start, age_end, location,
                                               event.time, event.step_size, self.randomness)
        self.population_view.update(population)

    @listens_for('time_step')
    def age_simulants(self, event):
        population = self.population_view.get(event.index, query="alive == 'alive'")
        population['age'] += event.step_size / pd.Timedelta(days=365)
        self.population_view.update(population)


class TestPopulation(NonCRNTestPopulation):
    def setup(self, builder):
        super().setup(builder)
        self.age_randomness = builder.randomness.get_stream('age_initialization', for_initialization=True)
        self.register = builder.randomness.register_simulants

    @listens_for('initialize_simulants', priority=0)
    def generate_test_population(self, event):
        age_start = event.user_data.get('age_start', self.config.population.age_start)
        age_end = event.user_data.get('age_end', self.config.population.age_end)
        age_draw = self.age_randomness.get_draw(event.index)
        if age_start == age_end:
            age = age_draw * (event.step_size / pd.Timedelta(days=365)) + age_start
        else:
            age = age_draw * (age_end - age_start) + age_start

        core_population = pd.DataFrame({'entrance_time': event.time,
                                        'age': age.values}, index=event.index)
        self.register(core_population)

        location = self.config.input_data.location_id
        population = _build_population(core_population, location, self.randomness)
        self.population_view.update(population)


def _build_population(core_population, location, randomness_stream):
    index = core_population.index

    population = pd.DataFrame(
        {'age': core_population['age'],
         'entrance_time': core_population['entrance_time'],
         'sex': randomness_stream.choice(index, ['Male', 'Female'], additional_key='sex_choice'),
         'alive': pd.Series('alive', index=index).astype(
             pd.api.types.CategoricalDtype(categories=['alive', 'dead', 'untracked'], ordered=False)),
         'location': location,
         'exit_time': pd.NaT, },
        index=index)
    return population


def _non_crn_build_population(index, age_start, age_end, location, event_time, step_size, randomness_stream):
    if age_start == age_end:
        age = randomness_stream.get_draw(index) * (step_size / pd.Timedelta(days=365)) + age_start
    else:
        age = randomness_stream.get_draw(index)*(age_end - age_start) + age_start

    population = pd.DataFrame(
        {'age': age,
         'sex': randomness_stream.choice(index, ['Male', 'Female'], additional_key='sex_choice'),
         'alive': pd.Series('alive', index=index).astype(
             pd.api.types.CategoricalDtype(categories=['alive', 'dead', 'untracked'], ordered=False)),
         'location': location,
         'entrance_time': event_time,
         'exit_time': pd.NaT, },
        index=index)
    return population


def make_dummy_column(name, initial_value):
    class dummy_column_maker:
        def setup(self, builder):
            self.population_view = builder.population.get_view([name])

        @listens_for('initialize_simulants')
        def make_column(self, event):
            self.population_view.update(pd.Series(initial_value, index=event.index, name=name))

        def __repr__(self):
            return f"dummy_column(name={name}, initial_value={initial_value})"
    return dummy_column_maker()




def get_randomness(key='test', clock=lambda: pd.Timestamp(1990, 7, 2), seed=12345, for_initialization=False):
    return randomness.RandomnessStream(key, clock, seed=seed, for_initialization=for_initialization)


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


def reset_mocks(mocks):
    for mock in mocks:
        mock.reset_mock()
