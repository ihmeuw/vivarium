import pytest

import pandas as pd
import numpy as np
import itertools

from vivarium.interpolation import Interpolation, validate_parameters, check_data_complete, make_left_edge, Order0Interp


def test_1d_interpolation():
    df = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100), 'c': np.arange(100, 0, -1)})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, (), ('a',), 1)

    query = pd.DataFrame({'a': np.arange(100, step=0.01)})

    assert np.allclose(query.a, i(query).b)
    assert np.allclose(100-query.a, i(query).c)


def test_age_year_interpolation():
    years = list(range(1990, 2010))
    ages = list(range(0, 90))
    pops = np.array(ages)*11.1
    data = []
    for age, pop in zip(ages, pops):
        for year in years:
            for sex in ['Male', 'Female']:
                data.append({'age': age, 'sex': sex, 'year': year, 'pop': pop})
    df = pd.DataFrame(data)

    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('sex', 'age'), ('year',), 1)
    query = pd.DataFrame({'year': [1990, 1990], 'age': [35, 35], 'sex': ['Male', 'Female']})
    assert np.allclose(i(query), 388.5)


def test_interpolation_called_missing_key_col():
    a = [range(1990, 1995), range(25, 30), ['Male', 'Female']]
    df = pd.DataFrame(list(itertools.product(*a)), columns=['year', 'age', 'sex'])
    df['pop'] = df.age * 11.1
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input
    i = Interpolation(df, ['sex',], ['year','age'], 1)
    query = pd.DataFrame({'year': [1990, 1990], 'age': [35, 35]})
    with pytest.raises(ValueError):
        i(query)


def test_interpolation_called_missing_param_col():
    a = [range(1990, 1995), range(25, 30), ['Male', 'Female']]
    df = pd.DataFrame(list(itertools.product(*a)), columns=['year', 'age', 'sex'])
    df['pop'] = df.age * 11.1
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input
    i = Interpolation(df, ['sex',], ['year','age'], 1)
    query = pd.DataFrame({'year': [1990, 1990], 'sex': ['Male', 'Female']})
    with pytest.raises(ValueError):
        i(query)


def test_2d_interpolation():
    a = np.mgrid[0:5, 0:5][0].reshape(25)
    b = np.mgrid[0:5, 0:5][1].reshape(25)
    df = pd.DataFrame({'a': a, 'b': b, 'c': b, 'd': a})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, (), ('a', 'b'), 1)

    query = pd.DataFrame({'a': np.arange(4, step=0.01), 'b': np.arange(4, step=0.01)})

    assert np.allclose(query.b, i(query).c)
    assert np.allclose(query.a, i(query).d)


def test_interpolation_with_categorical_parameters():
    a = ['one']*100 + ['two']*100
    b = np.append(np.arange(100), np.arange(100))
    c = np.append(np.arange(100), np.arange(100, 0, -1))
    df = pd.DataFrame({'a': a, 'b': b, 'c': c})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('a',), ('b',), 1)

    query_one = pd.DataFrame({'a': 'one', 'b': np.arange(100, step=0.01)})
    query_two = pd.DataFrame({'a': 'two', 'b': np.arange(100, step=0.01)})

    assert np.allclose(np.arange(100, step=0.01), i(query_one).c)

    assert np.allclose(np.arange(100, 0, step=-0.01), i(query_two).c)


def test_order_zero_2d():
    a = np.mgrid[0:5, 0:5][0].reshape(25)
    b = np.mgrid[0:5, 0:5][1].reshape(25)
    df = pd.DataFrame({'a': a + 0.5, 'b': b + 0.5, 'c': b*3, 'garbage': ['test']*len(a)})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('garbage',), ('a', 'b'), order=0)

    column = np.arange(4, step=0.011)
    query = pd.DataFrame({'a': column, 'b': column, 'garbage': ['test']*(len(column))})

    assert np.allclose(query.b.astype(int) * 3, i(query).c)


def test_order_zero_1d():
    s = pd.Series({0: 0, 1: 1}).reset_index()
    f = Interpolation(s, tuple(), ('index', ), order=0)

    df = pd.DataFrame({'index': [1, 0, 1, 1]}, index=[12, 3, 7, 2])
    f(df)

    assert f(pd.DataFrame({'index': [0]}))[0][0] == 0, 'should be precise at index values'
    assert f(pd.DataFrame({'index': [1]}))[0][0] == 1
    assert f(pd.DataFrame({'index': [2]}))[0][0] == 1, 'should be constant extrapolation outside of input range'
    assert f(pd.DataFrame({'index': [-1]}))[0][0] == 0


def test_validate_parameters__empty_data():
    with pytest.warns(UserWarning) as record:
        out, data, _ = validate_parameters(pd.DataFrame(columns=["age", "sex", "year", "value"]), ["sex"],
                                        ["age", "year"], 1)
    assert len(record) == 2
    message = record[0].message.args[0] + " " + record[1].message.args[0]
    assert "age" in message and "year" in message

    assert set(data.columns) == {"sex", "value"}


def test_check_data_complete_gaps():
    data = pd.DataFrame({'year_start': [1990, 1990, 1995, 1995],
                         'year_end': [1995, 1995, 2000, 2000],
                         'age_start': [16, 10, 10, 16],
                         'age_end': [20, 15, 15, 20],})

    with pytest.raises(NotImplementedError) as error:
        check_data_complete(data, [('year', 'year_start', 'year_end'), ['age', 'age_start', 'age_end']])

    message = error.value.args[0]

    assert "age_start" in message and "age_end" in message


def test_check_data_complete_overlap():
    data = pd.DataFrame({'year_start': [1995, 1995, 2000, 2005, 2010],
                         'year_end': [2000, 2000, 2005, 2010, 2015]})

    with pytest.raises(ValueError) as error:
        check_data_complete(data, [('year', 'year_start', 'year_end')])

    message = error.value.args[0]

    assert "year_start" in message and "year_end" in message


def test_check_data_missing_combos():
    data = pd.DataFrame({'year': [1990, 1990, 1995],
                         'age_start': [10, 15, 10],
                         'age_end': [15, 20, 15]})

    with pytest.raises(ValueError) as error:
        check_data_complete(data, ['year', ('age', 'age_start', 'age_end')])

    message = error.value.args[0]

    assert 'combination' in message


def test_make_left_edge():
    ages = [3, 7, 1]
    data = pd.DataFrame({'age': ages, 'year': [1990, 1992, 1991]})
    left_edge = make_left_edge(data, 'age')

    expected_left = [2, 5, 1]
    for i, a in enumerate(ages):
        assert left_edge[a] == expected_left[i]


def test_order0interp():
    data = pd.DataFrame({'year_start': [1990, 1990, 1990, 1990, 1995, 1995, 1995, 1995],
                         'year_end': [1995, 1995, 1995, 1995, 2000, 2000, 2000, 2000],
                         'age_start': [15, 10, 10, 15, 10, 10, 15, 15],
                         'age_end': [20, 15, 15, 20, 15, 15, 20, 20],
                         'height_start': [140, 160, 140, 160, 140, 160, 140, 160],
                         'height_end': [160, 180, 160, 180, 160, 180, 160, 180],
                         'value': [5, 3, 1, 7, 8, 6, 4, 2]})

    interp = Order0Interp(data, [('age', 'age_start', 'age_end'),
                                 ('year', 'year_start', 'year_end'),
                                 ('height', 'height_start', 'height_end'),]
                          , ['value'])

    interpolants = pd.DataFrame({'age': [12, 17, 8, 25, 12],
                                 'year': [1992, 1998, 1985, 1992, 1992],
                                 'height': [160, 145, 110, 185, 160]})

    result = interp(interpolants)
    assert result.equals(pd.DataFrame({'value': [3, 4, 1, 7, 3]}))


def test_order_zero_with_key_column():
    data = pd.DataFrame({'year_start': [1990, 1990, 1995, 1995],
                         'year_end': [1995, 1995, 2000, 2000],
                         'sex': ['Male', 'Female', 'Male', 'Female'],
                         'value_1': [10, 7, 2, 12],
                         'value_2': [1200, 1350, 1476, 1046]})

    i = Interpolation(data, ['sex',], [('year', 'year_start', 'year_end'),], 0)

    query = pd.DataFrame({'year': [1992, 1993,],
                          'sex': ['Male', 'Female']})

    expected_result = pd.DataFrame({'value_1': [10.0, 7.0],
                                    'value_2': [1200.0, 1350.0]})

    assert i(query).equals(expected_result)
