import numpy as np
import pandas as pd

from vivarium import InteractiveContext
from vivarium.testing_utilities import build_table, TestPopulation
from vivarium.framework.lookup import validate_parameters
from vivarium.framework.lookup import LookupTable
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
    base_config.update({'population': {'population_size': 10000},
                        'interpolation': {'order': 1}})  # the results we're checking later assume interp order 1

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(years, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None)
    ages = manager.build_table(ages, key_columns=('sex',), parameter_columns=('age', 'year',), value_columns=None)
    one_d_age = manager.build_table(one_d_age, key_columns=('sex',), parameter_columns=('age',), value_columns=None)

    pop = simulation.get_population(untracked=True)
    result_years = years(pop.index)
    result_ages = ages(pop.index)
    result_ages_1d = one_d_age(pop.index)

    fractional_year = simulation._clock.time.year
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, pop.age)
    assert np.allclose(result_ages_1d, pop.age)

    simulation._clock._time += pd.Timedelta(30.5 * 125, unit='D')
    simulation._population._population.age += 125/12

    result_years = years(pop.index)
    result_ages = ages(pop.index)
    result_ages_1d = one_d_age(pop.index)

    fractional_year = simulation._clock.time.year
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)
    assert np.allclose(result_ages, pop.age)
    assert np.allclose(result_ages_1d, pop.age)


@pytest.mark.skip(reason='only order 0 interpolation with age bin edges currently supported')
def test_interpolated_tables_without_uninterpolated_columns(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years = build_table(lambda age, sex, year: year, year_start, year_end)
    del years['sex']
    years = years.drop_duplicates()
    base_config.update({'population': {'population_size': 10000},
                        'interpolation': {'order': 1}})  # the results we're checking later assume interp order 1

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager.build_table(years, key_columns=(), parameter_columns=('year', 'age',), value_columns=None)

    result_years = years(simulation.get_population().index)

    fractional_year = simulation._clock.time.year
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)

    simulation._clock._time += pd.Timedelta(30.5 * 125, unit='D')

    result_years = years(simulation.get_population().index)

    fractional_year = simulation._clock.time.year
    fractional_year += simulation._clock.time.timetuple().tm_yday / 365.25

    assert np.allclose(result_years, fractional_year)


def test_interpolated_tables__exact_values_at_input_points(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    years = build_table(lambda age, sex, year: year, year_start, year_end)
    input_years = years.year_start.unique()
    base_config.update({'population': {'population_size': 10000}})

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    years = manager._build_table(years, key_columns=['sex'], parameter_columns=['age', 'year'], value_columns=None)

    for year in input_years:
        simulation._clock._time = pd.Timestamp(year, 1, 1)
        assert np.allclose(years(simulation.get_population().index),
                           simulation._clock.time.year + 1/365)


def test_lookup_table_scalar_from_list(base_config):
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = (manager._build_table((1, 2), key_columns=None, parameter_columns=None,
                                  value_columns=['a', 'b'])(simulation.get_population().index))

    assert isinstance(table, pd.DataFrame)
    assert table.columns.values.tolist() == ['a', 'b']
    assert np.all(table.a == 1)
    assert np.all(table.b == 2)


def test_lookup_table_scalar_from_single_value(base_config):
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = (manager._build_table(1, key_columns=None, parameter_columns=None,
                                  value_columns=['a'])(simulation.get_population().index))
    assert isinstance(table, pd.Series)
    assert np.all(table == 1)


def test_invalid_data_type_build_table(base_config):
    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    with pytest.raises(TypeError):
        manager._build_table('break', key_columns=None, parameter_columns=None, value_columns=None)


def test_lookup_table_interpolated_return_types(base_config):
    year_start = base_config.time.start.year
    year_end = base_config.time.end.year
    data = build_table(lambda age, sex, year: year, year_start, year_end)

    simulation = InteractiveContext(components=[TestPopulation()], configuration=base_config)
    manager = simulation._tables
    table = (manager._build_table(data, key_columns=['sex'], parameter_columns=['age', 'year'],
                                  value_columns=None)(simulation.get_population().index))
    # make sure a single value column is returned as a series
    assert isinstance(table, pd.Series)

    # now add a second value column to make sure the result is a df
    data['value2'] = data.value
    table = (manager._build_table(data, key_columns=['sex'],
                                  parameter_columns=['age', 'year'],
                                  value_columns=None)(simulation.get_population().index))

    assert isinstance(table, pd.DataFrame)


@pytest.mark.parametrize('data', [None, pd.DataFrame(), pd.DataFrame(columns=['a', 'b', 'c']), [], tuple()])
def test_validate_parameters_no_data(data):
    with pytest.raises(ValueError, match='supply some data'):
        validate_parameters(data, [], [], [])


@pytest.mark.parametrize('key_cols, param_cols, val_cols, match',
                         [(None, None, None, 'supply value_columns'),
                          (None, None, [], 'supply value_columns'),
                          (None, None, ['a', 'b'], 'match the number of values')])
def test_validate_parameters_error_scalar_data(key_cols, param_cols, val_cols, match):
    with pytest.raises(ValueError, match=match):
        validate_parameters([1, 2, 3], key_cols, param_cols, val_cols)


@pytest.mark.parametrize('key_cols, param_cols, val_cols, match',
                         [(['a', 'b'], ['b'], ['c'], 'no overlap'),
                          ([], ['b'], ['c'], 'do not match')])
def test_validate_parameters_error_dataframe(key_cols, param_cols, val_cols, match):
    data = pd.DataFrame({'a': [1, 2], 'b_start': [0, 5], 'b_end': [5, 10], 'c': [100, 150]})
    with pytest.raises(ValueError, match=match):
        validate_parameters(data, key_cols, param_cols, val_cols)


@pytest.mark.parametrize('data', ['FAIL', pd.Interval(5, 10), '2019-05-17', {'a': 5, 'b': 10}])
def test_validate_parameters_fail_other_data(data):
    with pytest.raises(TypeError, match='only allowable types'):
        validate_parameters(data, [], [], [])


@pytest.mark.parametrize('key_cols, param_cols, val_cols',
                         [(None, None, ['a', 'b', 'c']),
                          (None, ['d'], ['one', 'two', 'three']),
                          (['KEY'], None, ['a', 'b', 'c']),
                          (['KEY'], ['d'], ['a', 'b', 'c'])])
def test_validate_parameters_pass_scalar_data(key_cols, param_cols, val_cols):
    validate_parameters([1, 2, 3], key_cols, param_cols, val_cols)


@pytest.mark.parametrize('key_cols, param_cols, val_cols',
                         [(['a'], ['b'], ['c']),
                          ([], ['b'], ['c', 'a']),
                          ([], ['b'], ['a', 'c'])])
def test_validate_parameters_pass_dataframe(key_cols, param_cols, val_cols):
    data = pd.DataFrame({'a': [1, 2], 'b_start': [0, 5], 'b_end': [5, 10], 'c': [100, 150]})
    validate_parameters(data, key_cols, param_cols, val_cols)


@pytest.mark.parametrize('validate', [True, False])
def test_validate_option_invalid_data(validate):
    if validate:
        with pytest.raises(ValueError, match='supply some data'):
            lookup = LookupTable(0, [], None, [], [], [], 0, None, True, validate)
    else:
        lookup = LookupTable(0, [], None, [], [], [], 0, None, True, validate)


@pytest.mark.parametrize('validate', [True, False])
def test_validate_option_valid_data(validate):
    data = [1, 2, 3]
    key_cols = ['KEY']
    param_cols = ['d']
    val_cols = ['a', 'b', 'c']

    lookup = LookupTable(0, data, None, key_cols, param_cols, val_cols, 0, None, True, validate)
   