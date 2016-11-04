import pytest

import pandas as pd
import numpy as np

from ceam.interpolation import Interpolation

def test_1d_interpolation():
    df = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100), 'c': np.arange(100, 0, -1)})

    i = Interpolation(df, (), ('a',))

    query = pd.DataFrame({'a': np.arange(100, step=0.01)})

    assert np.allclose(query.a, i(query).b)
    assert np.allclose(100-query.a, i(query).c)

def test_2d_interpolation():
    a = np.mgrid[0:5,0:5][0].reshape(25)
    b = np.mgrid[0:5,0:5][1].reshape(25)
    df = pd.DataFrame({'a': a, 'b': b, 'c': b, 'd': a})

    i = Interpolation(df, (), ('a', 'b'))

    query = pd.DataFrame({'a': np.arange(4, step=0.01), 'b': np.arange(4, step=0.01)})

    assert np.allclose(query.b, i(query).c)
    assert np.allclose(query.a, i(query).d)

def test_interpolation_with_categorical_parameters():
    a = ['one']*100 + ['two']*100
    b = np.append(np.arange(100), np.arange(100))
    c = np.append(np.arange(100), np.arange(100, 0, -1))
    df = pd.DataFrame({'a': a, 'b': b, 'c': c})

    i = Interpolation(df, ('a',), ('b',))

    query_one = pd.DataFrame({'a': 'one', 'b': np.arange(100, step=0.01)})
    query_two = pd.DataFrame({'a': 'two', 'b': np.arange(100, step=0.01)})

    assert np.allclose(np.arange(100, step=0.01), i(query_one))

    assert np.allclose(np.arange(100, 0, step=-0.01), i(query_two))

def test_interpolation_with_function():
    df = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100), 'c': np.arange(100, 0, -1)})

    i = Interpolation(df, (), ('a',), func=lambda x: x * 2)

    query = pd.DataFrame({'a': np.arange(100, step=0.01)})

    assert np.allclose(query.a * 2, i(query).b)
