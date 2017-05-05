from datetime import datetime

import pytest
import pandas as pd
import numpy as np

import ceam.framework.randomness as random
from ceam.framework.randomness import RandomnessStream, RESIDUAL_CHOICE, RandomnessError


@pytest.fixture(params=[10**4, 10**5])
def index(request):
    return pd.Index(range(request.param)) if request.param else None


@pytest.fixture(params=[['a', 'small', 'bird']])
def choices(request):
    return request.param


# TODO: Add 2-d weights to the tests.
@pytest.fixture(params=[None, [10, 10, 10], [0.5, 0.1, 0.4]])
def weights(request):
    return request.param


@pytest.fixture(scope='function', params=[(0.2, 0.1, RESIDUAL_CHOICE),
                                          (0.5, 0.6, RESIDUAL_CHOICE),
                                          (.1, RESIDUAL_CHOICE, RESIDUAL_CHOICE)])
def weights_with_residuals(request):
    return request.param


def test_normalize_shape(weights, index):
    if weights is None:
        with pytest.raises(ValueError):
            random._normalize_shape(weights, index)
    else:
        p = random._normalize_shape(weights, index)
        assert p.shape == (len(index), len(weights))


def test__set_residual_probability(weights_with_residuals, index):
    # Coerce the weights to a 2-d numpy array.
    p = random._normalize_shape(weights_with_residuals, index)

    residual = np.where(p == RESIDUAL_CHOICE, 1, 0)
    non_residual = np.where(p != RESIDUAL_CHOICE, p, 0)

    if np.any(non_residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received un-normalized probability weights.
            random._set_residual_probability(p)

    elif np.any(residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received multiple instances of `RESIDUAL_CHOICE`
            random._set_residual_probability(p)

    else:  # Things should work
        p_total = np.sum(random._set_residual_probability(p))
        assert np.isclose(p_total, len(index), atol=0.0001)


def test_filter_for_probability(index):
    dates = [datetime(1991, 1, 1), datetime(1990, 1, 1)]
    randomness = RandomnessStream('test', dates.pop, 1)

    sub_index = randomness.filter_for_probability(index, 0.5)
    assert round(len(sub_index)/len(index), 1) == 0.5

    sub_sub_index = randomness.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index)/len(sub_index), 1) == 0.5


def test_choice(index, choices, weights):
    dates = [datetime(1990, 1, 1)]
    randomness = RandomnessStream('test', dates.pop, 1)

    chosen = randomness.choice(index, choices, p=weights)
    count = chosen.value_counts()
    # If we have weights, normalize them, otherwise generate uniform weights.
    weights = [w/sum(weights) for w in weights] if weights else [1/len(choices) for _ in choices]
    for k, c in count.items():
        assert np.isclose(c/len(index), weights[choices.index(k)], atol=0.01)


def test_choice_with_residuals(index, choices, weights_with_residuals):
    print(RESIDUAL_CHOICE in weights_with_residuals)
    dates = [datetime(1990, 1, 1)]
    randomness = RandomnessStream('test', dates.pop, 1)

    p = random._normalize_shape(weights_with_residuals, index)

    residual = np.where(p == RESIDUAL_CHOICE, 1, 0)
    non_residual = np.where(p != RESIDUAL_CHOICE, p, 0)

    if np.any(non_residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received un-normalized probability weights.
            randomness.choice(index, choices, p=weights_with_residuals)


    elif np.any(residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received multiple instances of `RESIDUAL_CHOICE`
            randomness.choice(index, choices, p=weights_with_residuals)

    else:  # Things should work
        chosen = randomness.choice(index, choices, p=weights_with_residuals)
        count = chosen.value_counts()
        print(weights_with_residuals)
        # We're relying on the fact that weights_with_residuals is a 1-d list
        resid_p = 1 - sum([w for w in weights_with_residuals if w != RESIDUAL_CHOICE])
        weights = [w if w != RESIDUAL_CHOICE else resid_p for w in weights_with_residuals]

        for k, c in count.items():
            assert np.isclose(c / len(index), weights[choices.index(k)], atol=0.01)








