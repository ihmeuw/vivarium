from datetime import datetime

import pandas as pd
import numpy as np

from ceam.framework.randomness import RandomnessStream, RESIDUAL_CHOICE


def test_filter_for_probability():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(10000))

    sub_index = r.filter_for_probability(index, 0.5)
    assert round(len(sub_index)/len(index), 1) == 0.5

    clock[0] = datetime(1991, 1, 1)

    sub_sub_index = r.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index)/len(sub_index), 1) == 0.5


def test_choice__default_weights():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(100000))

    chosen = r.choice(index, ['a', 'small', 'bird'])

    count = chosen.value_counts()
    for k, c in count.items():
        assert np.abs(c/len(index) - 1/3) < 0.005


def test_choice__homogenious_weights():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(100000))

    chosen = r.choice(index, ['a', 'small', 'bird'], [10, 10, 10])

    count = chosen.value_counts()
    for k, c in count.items():
        assert np.abs(c / len(index) - 1 / 3) < 0.005


def test_choice__hetrogenious_weights():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(100000))

    choices = ['a', 'small', 'bird']
    weights = [0.5, 0.1, 0.4]
    chosen = r.choice(index, choices, p=weights)

    count = chosen.value_counts()
    for c, p in zip(choices, weights):
        assert np.abs(count[c]/len(index) - p) < 0.005


def test_choice__residual_choice():
    clock = [datetime(1990, 1, 1)]
    r = RandomnessStream('test', lambda: clock[0], 1)

    index = pd.Index(range(100000))

    choices = ['a', 'small', 'bird']
    weights = [0.2, 0.1, RESIDUAL_CHOICE]
    chosen = r.choice(index, choices, p=weights)

    count = chosen.value_counts()
    expected_weights = [0.2, 0.1, 0.7]
    for c, p in zip(choices, expected_weights):
        assert np.abs(count[c]/len(index) - p) < 0.005
