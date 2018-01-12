"""This file has a ridiculous name."""
import pytest
import pandas as pd

from vivarium.test_util import TestPopulation, get_randomness, _build_population


@pytest.fixture(scope='function')
def pop_config(base_config):
    base_config.update(TestPopulation.configuration_defaults)
    return base_config


@pytest.fixture
def builder(pop_config):

    class BuilderMock:
        def __init__(self, config):
            self.configuration = config
            self.randomness = get_randomness

    return BuilderMock(pop_config)


@pytest.fixture
def test_pop(builder):
    pop = TestPopulation()
    pop.setup(builder)
    return pop


@pytest.fixture(params=[0, 5], ids=['single_age', 'age_range'])
def create_simulants_event(request):
    class Event:
        def __init__(self, index, time, user_data):
            self.index = index
            self.time = time
            self.step_size = pd.Timedelta(days=3)
            self.user_data = user_data

    return Event(index=pd.Index(range(10000)),
                 time=pd.Timestamp(year=2005, month=7, day=15),
                 user_data={'age_start': 0,
                            'age_end': request.param})


def test__build_population_age_distribution(test_pop, create_simulants_event):
    age_start = create_simulants_event.user_data.get('age_start', test_pop.config.population.age_start)
    age_end = create_simulants_event.user_data.get('age_end', test_pop.config.population.age_end)
    location = test_pop.config.input_data.location_id

    population = _build_population(create_simulants_event.index, age_start, age_end, location,
                                   create_simulants_event.time, create_simulants_event.step_size,
                                   test_pop.randomness)

    assert len(population) == len(population.age.unique())
    assert population.age.min() > age_start
    if age_start == age_end:
        assert population.age.max() < age_start + create_simulants_event.step_size.days/365.0
    else:
        assert population.age.max() < age_end
