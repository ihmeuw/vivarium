import numpy as np
import pandas as pd

from vivarium.testing_utilities import build_table, TestPopulation
from vivarium.interface.interactive import setup_simulation
import pytest


@pytest.mark.skip(reason='only order 0 interpolation with age bin edges currently supported')
def test_interpolated_tables(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years = build_table(lambda age, sex, year: year, year_start, year_end)
    ages = build_table(lambda age, sex, year: age, year_start, year_end)
    one_d_age = ages.copy()
    del one_d_age['year']
    one_d_age = one_d_age.drop_duplicates()
    base_config.population.update({'population_size': 10000,})
    base_config.interpolation.update({'order': 1})  # the results we're checking later assume interp order 1

    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    years = manager.build_table(years, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None)
    ages = manager.build_table(ages, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None)
    one_d_age = manager.build_table(one_d_age, key_columns=('sex',), parameter_columns=('age',), value_columns=None)

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_ages_1d = one_d_age(simulation.population.population.index)

    fractional_year = simulation.clock.time.year
    fractional_year += simulation.clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, simulation.population.population.age)
    assert np.allclose(result_ages_1d, simulation.population.population.age)

    simulation.clock._time += pd.Timedelta(30.5 * 125, unit='D')
    simulation.population._population.age += 125/12

    result_years = years(simulation.population.population.index)
    result_ages = ages(simulation.population.population.index)
    result_ages_1d = one_d_age(simulation.population.population.index)

    fractional_year = simulation.clock.time.year
    fractional_year += simulation.clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, simulation.population.population.age)
    assert np.allclose(result_ages_1d, simulation.population.population.age)


@pytest.mark.skip(reason='only order 0 interpolation with age bin edges currently supported')
def test_interpolated_tables_without_uninterpolated_columns(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years = build_table(lambda age, sex, year: year, year_start, year_end)
    del years['sex']
    years = years.drop_duplicates()
    base_config.population.update({'population_size': 10000})
    base_config.interpolation.update({'order': 1})  # the results we're checking later assume interp order 1

    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    years = manager.build_table(years, key_columns=(), parameter_columns=('year', 'age',), value_columns=None)

    result_years = years(simulation.population.population.index)

    fractional_year = simulation.clock.time.year
    fractional_year += simulation.clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)

    simulation.clock._time += pd.Timedelta(30.5 * 125, unit='D')

    result_years = years(simulation.population.population.index)

    fractional_year = simulation.clock.time.year
    fractional_year += simulation.clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)


def test_interpolated_tables__exact_values_at_input_points(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years = build_table(lambda age, sex, year: year, year_start, year_end)
    input_years = years.year_start.unique()
    base_config.population.update({'population_size': 10000})
    base_config.interpolation.update({'order': 0})

    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    years = manager.build_table(years, key_columns=('sex',),
                                parameter_columns=(['age', 'age_group_start', 'age_group_end'],
                                                   ['year', 'year_start', 'year_end'],),
                                value_columns=None)

    for year in input_years:
        simulation.clock._time = pd.Timestamp(year, 1, 1)
        assert np.allclose(years(simulation.population.population.index),
                           simulation.clock.time.year + 1/365)


def test_lookup_table_scalar_from_list(base_config):
    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    table = (manager.build_table((1,2), key_columns=None, parameter_columns=None,
                                 value_columns=['a', 'b'])(simulation.population.population.index))

    assert isinstance(table, pd.DataFrame)
    assert table.columns.values.tolist() == ['a', 'b']
    assert np.all(table.a == 1)
    assert np.all(table.b == 2)


def test_lookup_table_scalar_from_single_value(base_config):
    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    table = (manager.build_table(1, key_columns=None, parameter_columns=None,
                                 value_columns=['a'])(simulation.population.population.index))
    assert isinstance(table, pd.Series)
    assert np.all(table == 1)


def test_invalid_data_type_build_table(base_config):
    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    with pytest.raises(TypeError):
        manager.build_table('break', key_columns=None, parameter_columns=None, value_columns=None)


def test_lookup_table_interpolated_return_types(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    data = build_table(lambda age, sex, year: year, year_start, year_end)

    simulation = setup_simulation([TestPopulation()], input_config=base_config)
    manager = simulation.tables
    table = (manager.build_table(data, key_columns=('sex',),
                                 parameter_columns=[['age', 'age_group_start', 'age_group_end'],
                                                    ['year', 'year_start', 'year_end']],
                                 value_columns=None)(simulation.population.population.index))
    # make sure a single value column is returned as a series
    assert isinstance(table, pd.Series)

    # now add a second value column to make sure the result is a df
    data['value2'] = data.value
    table = (manager.build_table(data, key_columns=('sex',),
                                 parameter_columns=[['age', 'age_group_start', 'age_group_end'],
                                                    ['year', 'year_start', 'year_end']],
                                 value_columns=None)(simulation.population.population.index))

    assert isinstance(table, pd.DataFrame)

