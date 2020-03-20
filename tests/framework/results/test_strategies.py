import numpy as np
import pandas as pd
import pytest

from vivarium.framework.results import (MappingStrategy, BinningStrategy, MappingStrategyPool,
                                        Result, ResultProducerStrategy,
                                        FormattingStrategy)


@pytest.fixture
def data():
    size = 100
    data = pd.DataFrame({
        'color': np.random.choice(['red', 'blue', 'yellow'], size=size),
        'max_speed': np.random.uniform(50, 150, size=size),
        'design_date': pd.date_range('1-1-1990', '12-31-2020', periods=size)
    })
    return data


@pytest.fixture(params=[True, False])
def is_vectorized(request):
    return request.param


def check_columns(old_data, new_data, column):
    assert column not in old_data.columns
    assert column in new_data.columns
    assert isinstance(new_data[column].dtype, pd.CategoricalDtype)


def test_mapping_strategy_single_col_str_to_bool(data, is_vectorized):
    def is_red(color):
        return color == 'red'

    strategy = MappingStrategy('color', 'is_red', is_red, is_vectorized=is_vectorized)
    new_data = strategy(data)
    check_columns(data, new_data, 'is_red')
    assert (data.color == 'red').equals(new_data.is_red.astype(bool))


def test_mapping_strategy_single_col_float_to_bool(data, is_vectorized):
    def is_fast(max_speed):
        return max_speed > 120

    strategy = MappingStrategy('max_speed', 'is_fast', is_fast, is_vectorized=is_vectorized)
    new_data = strategy(data)
    check_columns(data, new_data, 'is_fast')
    assert (data.max_speed > 120).equals(new_data.is_fast.astype(bool))


def test_mapping_strategy_single_col_date_to_int(data, is_vectorized):
    def to_year(date):
        if is_vectorized:
            return date.dt.year
        else:
            return date.year

    strategy = MappingStrategy('design_date', 'design_year', to_year, is_vectorized=is_vectorized)
    new_data = strategy(data)
    check_columns(data, new_data, 'design_year')
    assert data.design_date.dt.year.equals(new_data.design_year.astype(int))


def test_mapping_strategy_multi_col_to_bool(data, is_vectorized):
    def is_dangerous(data_):
        return (data_['color'] == 'red') & (data_['max_speed'] > 120)

    strategy = MappingStrategy(['color', 'max_speed'], 'is_dangerous', is_dangerous, is_vectorized=is_vectorized)
    new_data = strategy(data)
    check_columns(data, new_data, 'is_dangerous')
    assert ((data.color == 'red') & (data.max_speed > 120)).equals(new_data.is_dangerous.astype(bool))


def test_binning_strategy_float(data):
    target = 'max_speed'
    binned_column = 'how_fast'
    bins = [0, 100, 1000]
    labels = ['slow', 'fast']

    strategy = BinningStrategy(target, binned_column, bins, labels)
    new_data = strategy(data)
    check_columns(data, new_data, 'how_fast')
    assert (data.max_speed <= 100).map({True: 'slow', False: 'fast'}).equals(new_data.how_fast.astype(str))


def test_binning_strategy_datetime(data):
    target = 'design_date'
    binned_column = 'when'
    bins = [pd.Timestamp('1-1-1990'), pd.Timestamp('1-1-2000'), pd.Timestamp('12-31-2020')]
    labels = ['ancient', 'modern']

    strategy = BinningStrategy(target, binned_column, bins, labels, include_lowest=True)
    new_data = strategy(data)
    check_columns(data, new_data, 'when')
    assert ((data.design_date <= pd.Timestamp('1-1-2000'))
            .map({True: 'ancient', False: 'modern'})
            .equals(new_data.when.astype(str)))


def test_mapping_strategy_pool_duplicate_result_column():
    strategy = MappingStrategy('coconut', 'unladen_swallow', lambda x: 'the spanish inquisition', True)
    pool = MappingStrategyPool()
    pool.add_strategy(strategy)
    with pytest.raises(ValueError, match='already exists'):
        pool.add_strategy(strategy)

    strategy2 = MappingStrategy('a herring', 'unladen_swallow', lambda x: 'the spanish inquisition', True)
    with pytest.raises(ValueError, match='already exists'):
        pool.add_strategy(strategy2)


def test_mapping_strategy_pool_single_strategy(data):
    target = 'max_speed'
    binned_column = 'how_fast'
    bins = [0, 100, 1000]
    labels = ['slow', 'fast']

    strategy = BinningStrategy(target, binned_column, bins, labels)
    pool = MappingStrategyPool()
    pool.add_strategy(strategy)
    assert strategy(data).equals(pool.expand_data(data))


def test_mapping_strategy_pool_multiple_strategies(data):
    target = 'max_speed'
    binned_column = 'how_fast'
    bins = [0, 100, 1000]
    labels = ['slow', 'fast']

    def is_dangerous(data_):
        return (data_['color'] == 'red') & (data_['max_speed'] > 120)

    strategy1 = BinningStrategy(target, binned_column, bins, labels)
    strategy2 = MappingStrategy(['color', 'max_speed'], 'is_dangerous', is_dangerous, is_vectorized=is_vectorized)
    pool = MappingStrategyPool()
    pool.add_strategy(strategy1)
    pool.add_strategy(strategy2)

    assert strategy1(strategy2(data)).equals(pool.expand_data(data))
    assert strategy2(strategy1(data)).equals(pool.expand_data(data))


def test_result_producer_strategy(data):
    measure = 'elderberries'
    strategy = ResultProducerStrategy(measure, lambda x: x.max_speed.sum(), {})
    data = data.drop(columns='design_date')
    r = strategy(data.groupby('color'))
    assert isinstance(r, Result)
    assert r.measure == measure
    assert r.data.name == 'value'
    assert np.allclose(r.data, data.groupby('color').max_speed.sum().sort_index())
    assert r.additional_keys == {}




def test_formatting_strategy_initialization():
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        FormattingStrategy()

