import pytest
import numpy as np
import pandas as pd

from vivarium.framework.utilities import (from_yearly, to_yearly, rate_to_probability, probability_to_rate,
                                          collapse_nested_dict, import_by_path)


def test_from_yearly():
    one_month = pd.Timedelta(days=30.5)
    rate = 0.01
    new_rate = from_yearly(rate, one_month)
    assert round(new_rate, 5) == round(0.0008356164383561645, 5)


def test_to_yearly():
    one_month = pd.Timedelta(days=30.5)
    rate = 0.0008356164383561645
    new_rate = to_yearly(rate, one_month)
    assert round(new_rate, 5) == round(0.01, 5)


def test_rate_to_probability():
    rate = 0.001
    prob = rate_to_probability(rate)
    assert round(prob, 5) == round(0.00099950016662497809, 5)


def test_probability_to_rate():
    prob = 0.00099950016662497809
    rate = probability_to_rate(prob)
    assert round(rate, 5) == round(0.001, 5)


def test_rate_to_probability_symmetry():
    rate = 0.0001
    for _ in range(100):
        prob = rate_to_probability(rate)
        assert round(rate, 5) == round(probability_to_rate(prob), 5)
        rate += (1-0.0001)/100.0


def test_rate_to_probability_vectorizability():
    rate = 0.001
    rate = np.array([rate]*100)
    prob = rate_to_probability(rate)
    assert round(prob[10], 5) == round(0.00099950016662497809, 5)
    assert round(np.sum(rate), 5) == round(np.sum(probability_to_rate(prob)), 5)


def test_collapse_nested_dict():
    source = {'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}
    result = collapse_nested_dict(source)
    assert set(result) == {
            ('a.b.c', 1),
            ('a.b.d', 2),
            ('e', 3),
            }


def test_import_class_by_path():
    cls = import_by_path('collections.abc.Set')
    from collections.abc import Set
    assert cls is Set


def test_import_function_by_path():
    func = import_by_path('vivarium.framework.utilities.import_by_path')
    assert func is import_by_path


def test_bad_import_by_path():
    with pytest.raises(ImportError):
        import_by_path('junk.garbage.SillyClass')
    with pytest.raises(AttributeError):
        import_by_path('vivarium.framework.components.SillyClass')
