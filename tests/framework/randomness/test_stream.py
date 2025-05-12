from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from layered_config_tree import LayeredConfigTree
from scipy import stats
from vivarium_testing_utils import FuzzyChecker

from tests.helpers import ColumnCreator
from vivarium import InteractiveContext
from vivarium.framework.randomness import RESIDUAL_CHOICE, RandomnessError, RandomnessStream
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import (
    PPFCallable,
    _normalize_shape,
    _set_residual_probability,
)


@pytest.fixture
def randomness_stream() -> RandomnessStream:
    dates = [pd.Timestamp(1991, 1, 1), pd.Timestamp(1990, 1, 1)]
    randomness = RandomnessStream("test", dates.pop, 1, IndexMap())
    return randomness


def test_normalize_shape(
    weights_with_residuals: tuple[Any, ...], index: pd.Index[int]
) -> None:
    p = _normalize_shape(weights_with_residuals, index)
    assert p.shape == (len(index), len(weights_with_residuals))


def test__set_residual_probability(
    weights_with_residuals: tuple[Any, ...], index: pd.Index[int]
) -> None:
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


def test_filter_for_probability_single_probability(
    randomness_stream: RandomnessStream, index: pd.Index[int]
) -> None:
    sub_index = randomness_stream.filter_for_probability(index, 0.5)
    assert np.isclose(len(sub_index) / len(index), 0.5, rtol=0.1)

    sub_sub_index = randomness_stream.filter_for_probability(sub_index, 0.5)
    assert np.isclose(len(sub_sub_index) / len(sub_index), 0.5, rtol=0.1)


def test_filter_for_probability_multiple_probabilities(
    randomness_stream: RandomnessStream, index: pd.Index[int]
) -> None:
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


@pytest.mark.parametrize(
    "rate, time_scaling_factor",
    [
        (0.5, 1),
        (0.25, 1.0),
        (0.5, 0.5),
        (0.25, 0.5),
    ],
)
def test_filter_for_rate_single_probability(
    randomness_stream: RandomnessStream,
    index: pd.Index[int],
    rate: float,
    time_scaling_factor: float,
    fuzzy_checker: FuzzyChecker,
) -> None:
    scaled_rate = rate * (time_scaling_factor / 365.0)
    sub_index = randomness_stream.filter_for_rate(index, scaled_rate)
    fuzzy_checker.fuzzy_assert_proportion(
        len(sub_index),
        len(index),
        scaled_rate,
    )

    sub_sub_index = randomness_stream.filter_for_rate(sub_index, scaled_rate)
    fuzzy_checker.fuzzy_assert_proportion(
        len(sub_sub_index),
        len(sub_index),
        scaled_rate,
    )


def test_filter_for_rate_multiple_probabilities(
    randomness_stream: RandomnessStream, index: pd.Index[int], fuzzy_checker: FuzzyChecker
) -> None:
    rates = pd.Series([0.3, 0.3, 0.3, 0.6, 0.6] * (index.size // 5), index=index)
    sub_index = randomness_stream.filter_for_rate(index, rates)
    fuzzy_checker.fuzzy_assert_proportion(
        len(sub_index),
        len(index),
        0.3 * (3 / 5) + 0.6 * (2 / 5),
    )


def test_choice(
    randomness_stream: RandomnessStream,
    index: pd.Index[int],
    choices: list[str],
    weights: None | list[float],
) -> None:
    chosen = randomness_stream.choice(index, choices, p=weights)  # type: ignore [arg-type]
    count = chosen.value_counts()
    # If we have weights, normalize them, otherwise generate uniform weights.
    weights = (
        [w / sum(weights) for w in weights]
        if weights
        else [1 / len(choices) for _ in choices]
    )
    for k, c in count.items():
        assert np.isclose(c / len(index), weights[choices.index(str(k))], atol=0.01)


def test_choice_with_residuals(
    randomness_stream: RandomnessStream,
    index: pd.Index[int],
    choices: list[str],
    weights_with_residuals: tuple[Any],
) -> None:
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
            assert np.isclose(c / len(index), weights[choices.index(str(k))], atol=0.01)


@pytest.mark.parametrize(
    "distribution, ppf, error_message",
    [
        (None, None, "Either distribution or ppf must be provided"),
        (stats.norm, lambda x: x, "Only one of distribution or ppf can be provided"),
    ],
)
def test_sample_from_distribution_bad_args(
    distribution: stats.rv_continuous | None,
    ppf: PPFCallable | None,
    error_message: str,
    randomness_stream: RandomnessStream,
) -> None:
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
    index: pd.Index[int], distribution: stats.rv_continuous, params: dict[str, int]
) -> None:
    randomness_stream = RandomnessStream(
        "test", lambda: pd.Timestamp(2020, 1, 1), 1, IndexMap()
    )
    draws = randomness_stream.get_draw(index, "some_key")
    expected = distribution.ppf(draws, **params)

    sample = randomness_stream.sample_from_distribution(
        index=index, distribution=distribution, ppf=None, additional_key="some_key", **params
    )

    assert isinstance(sample, pd.Series)
    assert sample.index.equals(index)
    assert np.allclose(sample, expected)


def test_sample_from_distribution_using_ppf(index: pd.Index[int]) -> None:
    def silly_ppf(x: pd.Series[Any], **kwargs: Any) -> pd.Series[Any]:
        add = kwargs["add"]
        mult = kwargs["mult"]
        output = mult * (x**2) + add
        assert isinstance(output, pd.Series)
        return output

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


@pytest.mark.parametrize(
    "rate_conversion",
    [
        "linear",
        "exponential",
        None,
    ],
)
def test_stream_rate_conversion_config(
    rate_conversion: str,
    base_config: LayeredConfigTree,
) -> None:
    cc = ColumnCreator()
    # Do not update key if key is not configured (None) to test default behavior
    if rate_conversion is not None:
        base_config.update(
            {"configuration": {"randomness": {"rate_conversion_type": rate_conversion}}}
        )
    sim = InteractiveContext(base_config, components=[cc])
    # Convert for default
    if rate_conversion is None:
        rate_conversion = "linear"
    assert sim._randomness._rate_conversion_type == rate_conversion
