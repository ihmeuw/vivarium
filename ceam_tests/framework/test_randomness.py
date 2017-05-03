from datetime import datetime

import pytest
import pandas as pd
import numpy as np

from ceam.framework.randomness import RandomnessStream, RESIDUAL_CHOICE, RandomnessError

TEST_POPULATION_SIZES = [10**4, 10**5]
TEST_CHOICES = [['a', 'small', 'bird']]
TEST_WEIGHTS = [None, [10, 10, 10], [0.5, 0.1, 0.4],
                [0.2, 0.1, RESIDUAL_CHOICE], [0.5, 0.6, RESIDUAL_CHOICE]]

@pytest.fixture
def randomness():
    dates = [datetime(1991, 1, 1), datetime(1990, 1, 1)]
    clock = dates.pop
    return RandomnessStream('test', clock, 1)


@pytest.fixture(params=TEST_POPULATION_SIZES)
def index(request):
    return pd.Index(range(request.param))


@pytest.fixture(params=TEST_CHOICES)
def choices(request):
    return request.param


@pytest.fixture(params=TEST_WEIGHTS)
def weights(request):
    return request.param


def test_filter_for_probability(randomness, index):
    sub_index = randomness.filter_for_probability(index, 0.5)
    assert round(len(sub_index)/len(index), 1) == 0.5

    sub_sub_index = randomness.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index)/len(sub_index), 1) == 0.5


def test_choice(randomness, index, choices, weights):
    if weights and RESIDUAL_CHOICE in weights and sum([w for w in weights if w != RESIDUAL_CHOICE]) > 1:
        with pytest.raises(RandomnessError):
            chosen = randomness.choice(index, choices, p=weights)
    else:
        chosen = randomness.choice(index, choices, p=weights)
        count = chosen.value_counts()
        weights = normalize(weights, len(choices))
        for k, c in count.items():
            assert np.abs(c/len(index) - weights[choices.index(k)]) < 0.01


def normalize(weights, length):
    if not weights:
        return [1/length for _ in range(length)]
    elif RESIDUAL_CHOICE in weights:
        weights[weights.index(RESIDUAL_CHOICE)] = 1 - sum([w for w in weights if w != RESIDUAL_CHOICE])
        return weights
    else:
        return [w/sum(weights) for w in weights]
