from itertools import chain, combinations, product

import pytest
import numpy as np
import pandas as pd
from scipy.stats import chisquare

from vivarium.framework.randomness import IndexMap, RandomnessError


def almost_powerset(iterable):
    """almost_powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"""
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(1, len(s)+1)))


def generate_keys(number, types=('int', 'float', 'datetime'), seed=123456):
    rs = np.random.RandomState(seed=seed)

    index = []
    if 'datetime' in types:
        year = rs.choice(np.arange(1980, 2018))
        day = rs.choice(pd.date_range(f'01/01/{year}', periods=365))
        start_time = rs.choice(pd.date_range(day, periods=86400, freq='s'))
        freq = rs.choice(['ms', 's', 'min', 'h'])
        index.append(pd.date_range(start_time, periods=number, freq=freq))

    if 'int' in types:
        kind = rs.choice(['random', 'sequential'])
        if kind == 'random':
            ints = np.unique(rs.randint(0, 1000 * number, size=100 * number))
            assert len(ints) > number
            rs.shuffle(ints)
            index.append(ints[:number])
        else:
            start = rs.randint(0, 100*number)
            index.append(np.arange(start, start + number, dtype=int))

    if 'float' in types:
        index.append(rs.random_sample(size=number))

    return pd.Series(0, index=index).index


rs = np.random.RandomState(seed=456789)
index_sizes = list(rs.randint(10_000, 250_000, size=1))
types = list(almost_powerset(['int', 'float', 'datetime']))
seeds = list(rs.randint(10000000, size=1))


def id_fun(param):
    return f"Size:{param[0]}, Types:{param[1]}, Seed:{param[2]}"


@pytest.fixture(scope='module', params=list(product(index_sizes, types, seeds)), ids=id_fun)
def map_size_and_hashed_values(request):
    keys = generate_keys(*request.param)
    m = IndexMap()
    return m.map_size, m.hash_(keys)


def test_digit_scalar():
    m = IndexMap()
    k = 123456789
    for i in range(10):
        assert m.digit(k, i) == 10 - (i + 1)


def test_digit_series():
    m = IndexMap()
    k = pd.Series(123456789, index=range(10000))
    for i in range(10):
        assert len(m.digit(k, i).unique()) == 1
        assert m.digit(k, i)[0] == 10 - (i + 1)


def test_clip_to_seconds_scalar():
    m = IndexMap()
    k = pd.to_datetime("2010-01-25 06:25:31.123456789")
    assert m.clip_to_seconds(k.value) == int(str(k.value)[:10])


def test_clip_to_seconds_series():
    m = IndexMap()
    stamp = 1234567890
    k = pd.date_range(pd.to_datetime(stamp, unit='s'), periods=10000, freq='ns').to_series().astype(np.int64)
    assert len(m.clip_to_seconds(k).unique()) == 1
    assert m.clip_to_seconds(k).unique()[0] == stamp


def test_spread_scalar():
    m = IndexMap()
    assert m.spread(1234567890) == 4072825790


def test_spread_series():
    m = IndexMap()
    s = pd.Series(1234567890, index=range(10000))
    assert len(m.spread(s).unique()) == 1
    assert m.spread(s).unique()[0] == 4072825790


def test_shift_scalar():
    m = IndexMap()
    assert m.shift(1.1234567890) == 1234567890


def test_shift_series():
    m = IndexMap()
    s = pd.Series(1.1234567890, index=range(10000))
    assert len(m.shift(s).unique()) == 1
    assert m.shift(s).unique()[0] == 1234567890


def test_convert_to_ten_digit_int():
    m = IndexMap()
    v = 1234567890
    datetime_col = pd.date_range(pd.to_datetime(v, unit='s'), periods=10000, freq='ns').to_series()
    int_col = pd.Series(v, index=range(10000))
    float_col = pd.Series(1.1234567890, index=range(10000))
    bad_col = pd.Series('a', index=range(10000))

    assert m.convert_to_ten_digit_int(datetime_col).unique()[0] == v
    assert m.convert_to_ten_digit_int(int_col).unique()[0] == 4072825790
    assert m.convert_to_ten_digit_int(float_col).unique()[0] == v
    with pytest.raises(RandomnessError):
        m.convert_to_ten_digit_int(bad_col)


@pytest.mark.skip("This fails because the hash needs work")
def test_hash_collisions(map_size_and_hashed_values):
    n, h = map_size_and_hashed_values
    k = len(h)

    expected_empty_bins = n*(1 - 1/n)**k  # Potential source of roundoff issues.
    expected_full_bins = n - expected_empty_bins
    expected_collisions = k - expected_full_bins

    uniques = h.drop_duplicates()
    actual_collisions = len(h) - len(uniques)

    assert actual_collisions/expected_collisions < 1.5, "Too many collisions"


@pytest.mark.skip("This fails because the hash needs work")
def test_hash_uniformity(map_size_and_hashed_values):
    n, h = map_size_and_hashed_values

    k = len(h)
    num_bins = k//5  # Want about 5 items per bin for chi-squared
    bins = np.linspace(0, n + 1, num_bins)

    binned_data = pd.cut(h, bins)
    distribution = pd.value_counts(binned_data).sort_index()
    c, p = chisquare(distribution)

    assert p > 0.05, "Data not uniform"


def test_update(mocker):
    m = IndexMap()
    keys = generate_keys(10000)

    def hash_mock(k, salt=0):
        seed = 123456
        rs = np.random.RandomState(seed=seed + salt)
        return pd.Series(rs.randint(0, len(k)*10, size=len(k)), index=k)

    mocker.patch.object(m, 'hash_', side_effect=hash_mock)
    m.update(keys)
    assert len(m) == len(keys), "All keys not in mapping"
    assert m._map.index.difference(keys).empty, "All keys not in mapping"
    assert len(m._map.unique()) == len(keys), "Duplicate values in mapping"

    # Can't have duplicate keys.
    with pytest.raises(KeyError):
        m.update(keys)

    new_unique_keys = generate_keys(1000).difference(keys)
    m.update(new_unique_keys)
    assert len(m) == len(keys) + len(new_unique_keys), "All keys not in mapping"
    assert m._map.index.difference(keys.union(new_unique_keys)).empty, "All keys not in mapping"
    assert len(m._map.unique()) == len(keys) + len(new_unique_keys), "Duplicate values in mapping"
