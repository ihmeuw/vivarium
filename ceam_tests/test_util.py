# ~/ceam/tests/test_util.py
import pytest

from datetime import timedelta
from unittest.mock import Mock

import numpy as np
import pandas as pd

from ceam.engine import SimulationModule
from ceam.util import from_yearly, to_yearly, rate_to_probability, probability_to_rate, filter_for_probability


# Simple regression tests for rate functions
def test_from_yearly():
    one_month = timedelta(days=30.5)
    rate = 0.01
    new_rate = from_yearly(rate, one_month)
    assert round(new_rate, 5) == round(0.0008356164383561645, 5)

def test_to_yearly():
    one_month = timedelta(days=30.5)
    rate = 0.0008356164383561645
    new_rate = to_yearly(rate, one_month)
    assert round(new_rate, 5) == round(0.01, 5)

def test_rate_to_probability():
    rate = 0.001
    prob = rate_to_probability(rate)
    assert round(prob, 5) == round(0.00099950016662497809, 5)

def test_probablity_to_rate():
    prob = 0.00099950016662497809
    rate = probability_to_rate(prob)
    assert round(rate, 5) == round(0.001, 5)

def test_rate_to_probability_symmetry():
    rate = 0.0001
    for _ in range(100):
        prob = rate_to_probability(rate)
        assert round(rate, 5) == round(probability_to_rate(prob), 5)
        rate += (1-0.0001)/100.0

def test_rate_to_probablity_vectorizability():
    rate = 0.001
    rate = np.array([rate]*100)
    prob = rate_to_probability(rate)
    assert round(prob[10], 5) == round(0.00099950016662497809, 5)
    assert round(np.sum(rate), 5) == round(np.sum(probability_to_rate(prob)), 5)

def test_filter_for_probability():
    pop = pd.DataFrame(dict(age=[0]*10000))

    sub_pop = filter_for_probability(pop, 0.5)
    assert round(len(sub_pop)/len(pop), 1) == 0.5

    sub_sub_pop = filter_for_probability(sub_pop, 0.5)
    assert round(len(sub_sub_pop)/len(sub_pop), 1) == 0.5
# End.
