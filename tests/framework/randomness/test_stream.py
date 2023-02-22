import numpy as np
import pandas as pd
import pytest

from vivarium.framework.randomness import (
    RESIDUAL_CHOICE,
    RandomnessError,
    RandomnessStream,
)
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import (
    _normalize_shape,
    _set_residual_probability,
)


@pytest.fixture
def randomness_stream():
    dates = [pd.Timestamp(1991, 1, 1), pd.Timestamp(1990, 1, 1)]
    randomness = RandomnessStream("test", dates.pop, 1, IndexMap())
    return randomness


def test_normalize_shape(weights_with_residuals, index):
    p = _normalize_shape(weights_with_residuals, index)
    assert p.shape == (len(index), len(weights_with_residuals))


def test__set_residual_probability(weights_with_residuals, index):
    # Coerce the weights to a 2-d numpy array.
    p = _normalize_shape(weights_with_residuals, index)

    residual = np.where(p == RESIDUAL_CHOICE, 1, 0)
    non_residual = np.where(p != RESIDUAL_CHOICE, p, 0)

    if np.any(non_residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received un-normalized probability weights.
            _set_residual_probability(p)

    elif np.any(residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received multiple instances of `RESIDUAL_CHOICE`
            _set_residual_probability(p)

    else:  # Things should work
        p_total = np.sum(_set_residual_probability(p))
        assert np.isclose(p_total, len(index), atol=0.0001)


def test_filter_for_probability(randomness_stream, index):

    sub_index = randomness_stream.filter_for_probability(index, 0.5)
    assert round(len(sub_index) / len(index), 1) == 0.5

    sub_sub_index = randomness_stream.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index) / len(sub_index), 1) == 0.5


def test_choice(randomness_stream, index, choices, weights):
    chosen = randomness_stream.choice(index, choices, p=weights)
    count = chosen.value_counts()
    # If we have weights, normalize them, otherwise generate uniform weights.
    weights = (
        [w / sum(weights) for w in weights]
        if weights
        else [1 / len(choices) for _ in choices]
    )
    for k, c in count.items():
        assert np.isclose(c / len(index), weights[choices.index(k)], atol=0.01)


def test_choice_with_residuals(randomness_stream, index, choices, weights_with_residuals):
    print(RESIDUAL_CHOICE in weights_with_residuals)

    p = _normalize_shape(weights_with_residuals, index)

    residual = np.where(p == RESIDUAL_CHOICE, 1, 0)
    non_residual = np.where(p != RESIDUAL_CHOICE, p, 0)

    if np.any(non_residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received un-normalized probability weights.
            randomness_stream.choice(index, choices, p=weights_with_residuals)

    elif np.any(residual.sum(axis=1) > 1):
        with pytest.raises(RandomnessError):
            # We received multiple instances of `RESIDUAL_CHOICE`
            randomness_stream.choice(index, choices, p=weights_with_residuals)

    else:  # Things should work
        chosen = randomness_stream.choice(index, choices, p=weights_with_residuals)
        count = chosen.value_counts()
        print(weights_with_residuals)
        # We're relying on the fact that weights_with_residuals is a 1-d list
        residual_p = 1 - sum([w for w in weights_with_residuals if w != RESIDUAL_CHOICE])
        weights = [w if w != RESIDUAL_CHOICE else residual_p for w in weights_with_residuals]

        for k, c in count.items():
            assert np.isclose(c / len(index), weights[choices.index(k)], atol=0.01)
