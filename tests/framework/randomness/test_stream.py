from typing import Dict

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from vivarium.framework.randomness import RESIDUAL_CHOICE, RandomnessError, RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import _normalize_shape, _set_residual_probability


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


def test_filter_for_probability_single_probability(randomness_stream, index):
    sub_index = randomness_stream.filter_for_probability(index, 0.5)
    assert np.isclose(len(sub_index) / len(index), 0.5, rtol=0.1)

    sub_sub_index = randomness_stream.filter_for_probability(sub_index, 0.5)
    assert np.isclose(len(sub_sub_index) / len(sub_index), 0.5, rtol=0.1)


def test_filter_for_probability_multiple_probabilities(randomness_stream, index):
    probabilities = pd.Series([0.3, 0.3, 0.3, 0.6, 0.6] * (index.size // 5), index=index)
    threshold_0_3 = probabilities.index[probabilities == 0.3]
    threshold_0_6 = probabilities.index.difference(threshold_0_3)

    sub_index = randomness_stream.filter_for_probability(index, probabilities)
    assert np.isclose(
        len(sub_index.intersection(threshold_0_3)) / len(threshold_0_3), 0.3, rtol=0.1
    )
    assert np.isclose(
        len(sub_index.intersection(threshold_0_6)) / len(threshold_0_6), 0.6, rtol=0.1
    )


def test_filter_for_rate_single_probability(randomness_stream, index):
    sub_index = randomness_stream.filter_for_rate(index, 0.5)
    assert np.isclose(len(sub_index) / len(index), 1 - np.exp(-0.5), rtol=0.1)

    sub_sub_index = randomness_stream.filter_for_rate(sub_index, 0.5)
    assert np.isclose(len(sub_sub_index) / len(sub_index), 1 - np.exp(-0.5), rtol=0.1)


def test_filter_for_rate_multiple_probabilities(randomness_stream, index):
    rates = pd.Series([0.3, 0.3, 0.3, 0.6, 0.6] * (index.size // 5), index=index)
    rate_0_3 = rates.index[rates == 0.3]
    rate_0_6 = rates.index.difference(rate_0_3)

    sub_index = randomness_stream.filter_for_rate(index, rates)
    assert np.isclose(
        len(sub_index.intersection(rate_0_3)) / len(rate_0_3), 1 - np.exp(-0.3), rtol=0.1
    )
    assert np.isclose(
        len(sub_index.intersection(rate_0_6)) / len(rate_0_6), 1 - np.exp(-0.6), rtol=0.1
    )


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


@pytest.mark.parametrize(
    "distribution, ppf, error_message",
    [
        (None, None, "Either distribution or ppf must be provided"),
        (stats.norm, lambda x: x, "Only one of distribution or ppf can be provided"),
    ],
)
def test_sample_from_distribution_bad_args(
    distribution, ppf, error_message, randomness_stream
):
    with pytest.raises(ValueError, match=error_message):
        randomness_stream.sample_from_distribution(
            index=pd.Index([]),
            distribution=distribution,
            ppf=ppf,
        )


@pytest.mark.parametrize(
    "distribution, params",
    [
        (stats.norm, {"loc": 5, "scale": 1}),
        (stats.beta, {"a": 1, "b": 2}),
    ],
)
def test_sample_from_distribution_using_scipy(
    index: pd.Index, distribution: stats.rv_continuous, params: Dict
):
    randomness_stream = RandomnessStream(
        "test", lambda: pd.Timestamp(2020, 1, 1), 1, IndexMap()
    )
    draws = randomness_stream.get_draw(index, "some_key")
    expected = distribution.ppf(draws, **params)

    sample = randomness_stream.sample_from_distribution(
        index=index, distribution=distribution, additional_key="some_key", **params
    )

    assert isinstance(sample, pd.Series)
    assert sample.index.equals(index)
    assert np.allclose(sample, expected)


def test_sample_from_distribution_using_ppf(index: pd.Index):
    def silly_ppf(x, add, mult):
        return mult * (x**2) + add

    randomness_stream = RandomnessStream(
        "test", lambda: pd.Timestamp(2020, 1, 1), 1, IndexMap()
    )
    draws = randomness_stream.get_draw(index, "some_key")
    expected = 2 * (draws**2) + 1

    sample = randomness_stream.sample_from_distribution(
        index=index, ppf=silly_ppf, additional_key="some_key", add=1, mult=2
    )

    assert isinstance(sample, pd.Series)
    assert sample.index.equals(index)
    assert np.allclose(sample, expected)
