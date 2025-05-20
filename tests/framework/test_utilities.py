from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vivarium.framework.utilities import (
    collapse_nested_dict,
    from_yearly,
    handle_exceptions,
    import_by_path,
    probability_to_rate,
    rate_to_probability,
    to_yearly,
)


def test_from_yearly() -> None:
    one_month = pd.Timedelta(days=30.5)
    rate = 0.01
    new_rate = from_yearly(rate, one_month)
    assert isinstance(new_rate, float)
    assert round(new_rate, 5) == round(0.0008356164383561645, 5)


def test_to_yearly() -> None:
    one_month = pd.Timedelta(days=30.5)
    rate = 0.0008356164383561645
    new_rate = to_yearly(rate, one_month)
    assert isinstance(new_rate, float)
    assert round(new_rate, 5) == round(0.01, 5)


@pytest.mark.parametrize(
    "rate, time_scaling_factor",
    [
        (0.5, 1),
        (0.25, 1.0),
        (0.5, 0.5),
        (0.25, 0.5),
    ],
)
def test_rate_to_probability(rate: float, time_scaling_factor: float) -> None:
    expected = rate * time_scaling_factor
    prob = rate_to_probability(rate, time_scaling_factor)
    expected == prob


@pytest.mark.parametrize("rate", [0.75, pd.Series([0.75, 0.5, 0.25]), 0.1])
def test_rate_to_probability_default_time_scaling(rate: float) -> None:
    expected = rate
    prob = rate_to_probability(rate)
    assert (prob == expected).all()


def test_very_high_rate_to_probability() -> None:
    rate = np.array([10_000])
    prob = rate_to_probability(rate)
    assert np.isclose(prob, 1.0)


@pytest.mark.parametrize(
    "prob, time_scaling_factor",
    [
        (0.5, 1.0),
        (0.25, 1.0),
        (0.5, 0.5),
        (0.25, 0.5),
    ],
)
def test_probability_to_rate(prob: float, time_scaling_factor: float) -> None:
    expected = prob / time_scaling_factor
    probability = np.array([prob])
    rate = probability_to_rate(probability, time_scaling_factor)
    (rate == expected).all()


@pytest.mark.parametrize("prob", [0.1, pd.Series([0.2, 0.4, 0.6, 0.8]), 0.9])
def test_probability_to_rate_default_time_scaling(prob: float) -> None:
    expected = prob
    rate = probability_to_rate(prob)
    if isinstance(expected, float):
        rate == expected
    else:
        assert (rate == expected).all()


def test_rate_to_probability_symmetry() -> None:
    rate = np.array([0.0001])
    for _ in range(100):
        prob = rate_to_probability(rate)
        assert np.isclose(rate, probability_to_rate(prob))
        rate += (1 - 0.0001) / 100.0


def test_rate_to_probability_vectorizability() -> None:
    rate = 0.001
    rate_array = np.array([rate] * 100)
    prob_array = rate_to_probability(rate_array)
    sum_of_rates = np.sum(probability_to_rate(prob_array))

    assert isinstance(prob_array[10], float)
    assert round(prob_array[10], 5) == round(0.00099950016662497809, 5)
    assert isinstance(sum_of_rates, float)
    assert round(np.sum(rate_array), 5) == round(sum_of_rates, 5)


def test_collapse_nested_dict() -> None:
    source = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
    result = collapse_nested_dict(source)
    assert set(result) == {
        ("a.b.c", 1),
        ("a.b.d", 2),
        ("e", 3),
    }


def test_import_class_by_path() -> None:
    cls = import_by_path("collections.abc.Set")
    from collections.abc import Set

    assert cls is Set


def test_import_function_by_path() -> None:
    func = import_by_path("vivarium.framework.utilities.import_by_path")
    assert func is import_by_path


def test_bad_import_by_path() -> None:
    with pytest.raises(ImportError):
        import_by_path("junk.garbage.SillyClass")
    with pytest.raises(AttributeError):
        import_by_path("vivarium.framework.components.SillyClass")


class CustomException(Exception):
    pass


@pytest.mark.parametrize("test_input", [KeyboardInterrupt, RuntimeError, CustomException])
def test_handle_exceptions(test_input: type[BaseException]) -> None:
    def raise_me(ex: type[BaseException]) -> None:
        raise ex()

    with pytest.raises(test_input):
        # known issue with mypy
        # see heated thread at https://github.com/python/mypy/issues/6549
        func = handle_exceptions(raise_me(test_input), None, False)  # type: ignore[func-returns-value]
        func()


@pytest.mark.parametrize(
    "rate",
    [
        150.0,
        pd.Series([250.0, 300.0, 0.5, 0.25]),
    ],
)
def test_rate_to_probability_clipped(
    rate: float | pd.Series[float], caplog: pytest.LogCaptureFixture
) -> None:
    prob = rate_to_probability(rate)
    assert prob.max() <= 1.0
    if isinstance(rate, float):
        assert (prob == pd.Series([1.0])).all()
    else:
        assert (prob == pd.Series([1.0, 1.0, 0.5, 0.25])).all()
    assert "The probability has been clipped to 1.0" in caplog.text
