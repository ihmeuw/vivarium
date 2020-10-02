import pytest

import pandas as pd
import numpy as np
import itertools

from vivarium.interpolation import Interpolation, validate_parameters, check_data_complete, Order0Interp


def make_bin_edges(data: pd.DataFrame, col: str) -> pd.DataFrame:
    """ Given a dataframe and a column containing midpoints, construct
    equally sized bins around midpoints.
    """
    mid_pts = data[[col]].drop_duplicates().sort_values(by=col).reset_index(drop=True)
    mid_pts['shift'] = mid_pts[col].shift()

    mid_pts['left'] = mid_pts.apply(lambda row: (row[col] if pd.isna(row['shift'])
                                                 else 0.5 * (row[col] + row['shift'])), axis=1)

    mid_pts['right'] = mid_pts['left'].shift(-1)
    mid_pts['right'] = mid_pts.right.fillna(mid_pts.right.max() + mid_pts.left.tolist()[-1] - mid_pts.left.tolist()[-2])

    data = data.copy()
    idx = data.index

    data = data.set_index(col, drop=False)
    mid_pts = mid_pts.set_index(col, drop=False)

    data[[col, f'{col}_left', f'{col}_right']] = mid_pts[[col, 'left', 'right']]

    return data.set_index(idx)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_1d_interpolation():
    df = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100), 'c': np.arange(100, 0, -1)})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, (), ('a',), 1, True)

    query = pd.DataFrame({'a': np.arange(100, step=0.01)})

    assert np.allclose(query.a, i(query).b)
    assert np.allclose(100-query.a, i(query).c)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
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

    i = Interpolation(df, ('sex', 'age'), ('year',), 1, True)
    query = pd.DataFrame({'year': [1990, 1990], 'age': [35, 35], 'sex': ['Male', 'Female']})
    assert np.allclose(i(query), 388.5)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolation_called_missing_key_col():
    a = [range(1990, 1995), range(25, 30), ['Male', 'Female']]
    df = pd.DataFrame(list(itertools.product(*a)), columns=['year', 'age', 'sex'])
    df['pop'] = df.age * 11.1
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input
    i = Interpolation(df, ['sex',], ['year','age'], 1, True)
    query = pd.DataFrame({'year': [1990, 1990], 'age': [35, 35]})
    with pytest.raises(ValueError):
        i(query)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolation_called_missing_param_col():
    a = [range(1990, 1995), range(25, 30), ['Male', 'Female']]
    df = pd.DataFrame(list(itertools.product(*a)), columns=['year', 'age', 'sex'])
    df['pop'] = df.age * 11.1
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input
    i = Interpolation(df, ['sex',], ['year','age'], 1, True)
    query = pd.DataFrame({'year': [1990, 1990], 'sex': ['Male', 'Female']})
    with pytest.raises(ValueError):
        i(query)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_2d_interpolation():
    a = np.mgrid[0:5, 0:5][0].reshape(25)
    b = np.mgrid[0:5, 0:5][1].reshape(25)
    df = pd.DataFrame({'a': a, 'b': b, 'c': b, 'd': a})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, (), ('a', 'b'), 1, True)

    query = pd.DataFrame({'a': np.arange(4, step=0.01), 'b': np.arange(4, step=0.01)})

    assert np.allclose(query.b, i(query).c)
    assert np.allclose(query.a, i(query).d)


@pytest.mark.skip(reason="only order 0 interpolation currently supported")
def test_interpolation_with_categorical_parameters():
    a = ['one']*100 + ['two']*100
    b = np.append(np.arange(100), np.arange(100))
    c = np.append(np.arange(100), np.arange(100, 0, -1))
    df = pd.DataFrame({'a': a, 'b': b, 'c': c})
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('a',), ('b',), 1, True)

    query_one = pd.DataFrame({'a': 'one', 'b': np.arange(100, step=0.01)})
    query_two = pd.DataFrame({'a': 'two', 'b': np.arange(100, step=0.01)})

    assert np.allclose(np.arange(100, step=0.01), i(query_one).c)

    assert np.allclose(np.arange(100, 0, step=-0.01), i(query_two).c)


def test_order_zero_2d():
    a = np.mgrid[0:5, 0:5][0].reshape(25)
    b = np.mgrid[0:5, 0:5][1].reshape(25)
    df = pd.DataFrame({'a': a + 0.5, 'b': b + 0.5, 'c': b*3, 'garbage': ['test']*len(a)})
    df = make_bin_edges(df, 'a')
    df = make_bin_edges(df, 'b')
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('garbage',), [('a', 'a_left', 'a_right'), ('b', 'b_left', 'b_right')],
                      order=0, extrapolate=True, validate=True)

    column = np.arange(0.5, 4, step=0.011)
    query = pd.DataFrame({'a': column, 'b': column, 'garbage': ['test']*(len(column))})

    assert np.allclose(query.b.astype(int) * 3, i(query).c)


def test_order_zero_2d_fails_on_extrapolation():
    a = np.mgrid[0:5, 0:5][0].reshape(25)
    b = np.mgrid[0:5, 0:5][1].reshape(25)
    df = pd.DataFrame({'a': a + 0.5, 'b': b + 0.5, 'c': b*3, 'garbage': ['test']*len(a)})
    df = make_bin_edges(df, 'a')
    df = make_bin_edges(df, 'b')
    df = df.sample(frac=1)  # Shuffle table to assure interpolation works given unsorted input

    i = Interpolation(df, ('garbage',), [('a', 'a_left', 'a_right'), ('b', 'b_left', 'b_right')],
                      order=0, extrapolate=False, validate=True)

    column = np.arange(4, step=0.011)
    query = pd.DataFrame({'a': column, 'b': column, 'garbage': ['test']*(len(column))})

    with pytest.raises(ValueError) as error:
        i(query)

    message = error.value.args[0]

    assert 'Extrapolation' in message and 'a' in message


def test_order_zero_1d_no_extrapolation():
    s = pd.Series({0: 0, 1: 1}).reset_index()
    s = make_bin_edges(s, 'index')
    f = Interpolation(s, tuple(), [['index', 'index_left', 'index_right']], order=0, extrapolate=False,
                      validate=True)

    assert f(pd.DataFrame({'index': [0]}))[0][0] == 0, 'should be precise at index values'
    assert f(pd.DataFrame({'index': [0.999]}))[0][0] == 1

    with pytest.raises(ValueError) as error:
        f(pd.DataFrame({'index': [1]}))

    message = error.value.args[0]
    assert 'Extrapolation' in message and 'index' in message


def test_order_zero_1d_constant_extrapolation():
    s = pd.Series({0: 0, 1: 1}).reset_index()
    s = make_bin_edges(s, 'index')
    f = Interpolation(s, tuple(), [['index', 'index_left', 'index_right']], order=0, extrapolate=True,
                      validate=True)

    assert f(pd.DataFrame({'index': [1]}))[0][0] == 1
    assert f(pd.DataFrame({'index': [2]}))[0][0] == 1, 'should be constant extrapolation outside of input range'
    assert f(pd.DataFrame({'index': [-1]}))[0][0] == 0


def test_validate_parameters__empty_data():
    with pytest.raises(ValueError) as error:
        validate_parameters(pd.DataFrame(columns=["age_left", "age_right",
                                                  "sex", "year_left", "year_right", "value"]), ["sex"],
                            [("age", "age_left", "age_right"),
                             ["year", "year_left", "year_right"]])
    message = error.value.args[0]
    assert 'empty' in message


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
    data = pd.DataFrame({'year_start': [1990, 1990, 1995],
                         'year_end': [1995, 1995, 2000],
                         'age_start': [10, 15, 10],
                         'age_end': [15, 20, 15]})

    with pytest.raises(ValueError) as error:
        check_data_complete(data, [['year', 'year_start', 'year_end'], ('age', 'age_start', 'age_end')])

    message = error.value.args[0]

    assert 'combination' in message


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
                          , ['value'], True, True)

    interpolants = pd.DataFrame({'age': [12, 17, 8, 24, 12],
                                 'year': [1992, 1998, 1985, 1992, 1992],
                                 'height': [160, 145, 140, 179, 160]})

    result = interp(interpolants)
    assert result.equals(pd.DataFrame({'value': [3, 4, 1, 7, 3]}))


def test_order_zero_1d_with_key_column():
    data = pd.DataFrame({'year_start': [1990, 1990, 1995, 1995],
                         'year_end': [1995, 1995, 2000, 2000],
                         'sex': ['Male', 'Female', 'Male', 'Female'],
                         'value_1': [10, 7, 2, 12],
                         'value_2': [1200, 1350, 1476, 1046]})

    i = Interpolation(data, ['sex',], [('year', 'year_start', 'year_end'),], 0, True, True)

    query = pd.DataFrame({'year': [1992, 1993,],
                          'sex': ['Male', 'Female']})

    expected_result = pd.DataFrame({'value_1': [10.0, 7.0],
                                    'value_2': [1200.0, 1350.0]})

    assert i(query).equals(expected_result)


def test_order_zero_non_numeric_values():
    data = pd.DataFrame({'year_start': [1990, 1990],
                         'year_end': [1995, 1995],
                         'age_start': [15, 24,],
                         'age_end': [24, 30],
                         'value_1': ['blue', 'red']})

    i = Interpolation(data, tuple(), [('year', 'year_start', 'year_end'), ('age', 'age_start', 'age_end')], 0,
                      True, True)

    query = pd.DataFrame({'year': [1990, 1990],
                          'age': [15, 24,]},
                         index=[1, 0])

    expected_result = pd.DataFrame({'value_1': ['blue', 'red']},
                                   index=[1, 0])

    assert i(query).equals(expected_result)


def test_order_zero_3d_with_key_col():
    data = pd.DataFrame({'year_start': [1990, 1990, 1990, 1990, 1995, 1995, 1995, 1995]*2,
                         'year_end': [1995, 1995, 1995, 1995, 2000, 2000, 2000, 2000]*2,
                         'age_start': [15, 10, 10, 15, 10, 10, 15, 15]*2,
                         'age_end': [20, 15, 15, 20, 15, 15, 20, 20]*2,
                         'height_start': [140, 160, 140, 160, 140, 160, 140, 160]*2,
                         'height_end': [160, 180, 160, 180, 160, 180, 160, 180]*2,
                         'sex': ['Male']*8+['Female']*8,
                         'value': [5, 3, 1, 7, 8, 6, 4, 2, 6, 4, 2, 8, 9, 7, 5, 3]})

    interp = Interpolation(data, ('sex',),
                           [('age', 'age_start', 'age_end'),
                            ('year', 'year_start', 'year_end'),
                            ('height', 'height_start', 'height_end')], 0, True, True)

    interpolants = pd.DataFrame({'age': [12, 17, 8, 24, 12],
                                 'year': [1992, 1998, 1985, 1992, 1992],
                                 'height': [160, 145, 140, 185, 160],
                                 'sex': ['Male', 'Female', 'Female', 'Male', 'Male']},
                                index=[10, 4, 7, 0, 9])

    result = interp(interpolants)
    assert result.equals(pd.DataFrame({'value': [3.0, 5.0, 2.0, 7.0, 3.0]}, index=[10, 4, 7, 0, 9]))


def test_order_zero_diff_bin_sizes():
    data = pd.DataFrame({'year_start': [1990, 1995, 1996, 2005, 2005.5,],
                         'year_end': [1995, 1996, 2005, 2005.5, 2010],
                         'value': [1, 5, 2.3, 6, 100]})

    i = Interpolation(data, tuple(), [('year', 'year_start', 'year_end')], 0, False, True)

    query = pd.DataFrame({'year': [2007, 1990, 2005.4, 1994, 2004, 1995, 2002, 1995.5, 1996]})

    expected_result = pd.DataFrame({'value': [100, 1, 6, 1, 2.3, 5, 2.3, 5, 2.3]})

    assert i(query).equals(expected_result)


def test_order_zero_given_call_column():
    data = pd.DataFrame({'year_start': [1990, 1995, 1996, 2005, 2005.5,],
                         'year_end': [1995, 1996, 2005, 2005.5, 2010],
                         'year': [1992.5, 1995.5, 2000, 2005.25, 2007.75],
                         'value': [1, 5, 2.3, 6, 100]})

    i = Interpolation(data, tuple(), [('year', 'year_start', 'year_end')], 0, False, True)

    query = pd.DataFrame({'year': [2007, 1990, 2005.4, 1994, 2004, 1995, 2002, 1995.5, 1996]})

    expected_result = pd.DataFrame({'value': [100, 1, 6, 1, 2.3, 5, 2.3, 5, 2.3]})

    assert i(query).equals(expected_result)


@pytest.mark.parametrize('validate', [True, False])
def test_interpolation_init_validate_option_invalid_data(validate):
    if validate:
        with pytest.raises(ValueError, match='You must supply non-empty data to create the interpolation.'):
            i = Interpolation(pd.DataFrame(),[],[],0,True,validate)
    else:
        i = Interpolation(pd.DataFrame(),[],[],0,True,validate)


@pytest.mark.parametrize('validate', [True, False])
def test_interpolation_init_validate_option_valid_data(validate):
    s = pd.Series({0: 0, 1: 1}).reset_index()
    s = make_bin_edges(s, 'index')
    i = Interpolation(s, tuple(), [['index', 'index_left', 'index_right']], 0, True, validate)


@pytest.mark.parametrize('validate', [True, False])
def test_interpolation_call_validate_option_invalid_data(validate):
    s = pd.Series({0: 0, 1: 1}).reset_index()
    s = make_bin_edges(s, 'index')
    i = Interpolation(s, tuple(), [['index', 'index_left', 'index_right']], 0, True, validate)
    if validate:
        with pytest.raises(TypeError, match=r'Interpolations can only be called on pandas.DataFrames.*'):
            result = i(1)
    else:
        with pytest.raises(AttributeError):
            result = i(1)

@pytest.mark.parametrize('validate', [True, False])
def test_interpolation_call_validate_option_valid_data(validate):
    data = pd.DataFrame({'year_start': [1990, 1995, 1996, 2005, 2005.5,],
                         'year_end': [1995, 1996, 2005, 2005.5, 2010],
                         'value': [1, 5, 2.3, 6, 100]})

    i = Interpolation(data, tuple(), [('year', 'year_start', 'year_end')], 0, False, validate)
    query = pd.DataFrame({'year': [2007, 1990, 2005.4, 1994, 2004, 1995, 2002, 1995.5, 1996]})
    
    result = i(query)


@pytest.mark.parametrize('validate', [True, False])
def test_order0interp_validate_option_invalid_data(validate):
    data = pd.DataFrame({'year_start': [1995, 1995, 2000, 2005, 2010],
                         'year_end': [2000, 2000, 2005, 2010, 2015]})
   
    if validate:
        with pytest.raises(ValueError) as error:
            interp = Order0Interp(data, [('year', 'year_start', 'year_end')], [], True, validate)
            message = error.value.args[0]
            assert "year_start" in message and "year_end" in message
    else:
        interp = Order0Interp(data, [('year', 'year_start', 'year_end')], [], True, validate)


@pytest.mark.parametrize('validate', [True, False])
def test_order0interp_validate_option_valid_data(validate):
    data = pd.DataFrame({'year_start': [1990, 1995],
                         'year_end': [1995, 2000],
                         'value': [5, 3]})

    interp = Order0Interp(data, [('year', 'year_start', 'year_end')], ['value'], True, validate)
   