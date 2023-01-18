import numpy as np
import pandas as pd
import pytest

from vivarium.framework.randomness import (
    RESIDUAL_CHOICE,
    RandomnessError,
    RandomnessStream,
)
from vivarium.framework.randomness import core as random


def test_filter_for_probability(index):
    dates = [pd.Timestamp(1991, 1, 1), pd.Timestamp(1990, 1, 1)]
    randomness = RandomnessStream("test", dates.pop, 1)

    sub_index = randomness.filter_for_probability(index, 0.5)
    assert round(len(sub_index) / len(index), 1) == 0.5

    sub_sub_index = randomness.filter_for_probability(sub_index, 0.5)
    assert round(len(sub_sub_index) / len(sub_index), 1) == 0.5


def test_choice(index, choices, weights):
    dates = [pd.Timestamp(1990, 1, 1)]
    randomness = RandomnessStream("test", dates.pop, 1)

    chosen = randomness.choice(index, choices, p=weights)
    count = chosen.value_counts()
    # If we have weights, normalize them, otherwise generate uniform weights.
    weights = (
        [w / sum(weights) for w in weights]
        if weights
        else [1 / len(choices) for _ in choices]
    )
    for k, c in count.items():
        assert np.isclose(c / len(index), weights[choices.index(k)], atol=0.01)


def test_choice_with_residuals(index, choices, weights_with_residuals):
    print(RESIDUAL_CHOICE in weights_with_residuals)
    dates = [pd.Timestamp(1990, 1, 1)]
    randomness = RandomnessStream("test", dates.pop, 1)

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
        residual_p = 1 - sum([w for w in weights_with_residuals if w != RESIDUAL_CHOICE])
        weights = [w if w != RESIDUAL_CHOICE else residual_p for w in weights_with_residuals]

        for k, c in count.items():
            assert np.isclose(c / len(index), weights[choices.index(k)], atol=0.01)
