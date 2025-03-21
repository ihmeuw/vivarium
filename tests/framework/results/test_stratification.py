from __future__ import annotations

import re
from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from pytest_mock import MockerFixture

from tests.framework.results.helpers import (
    HOUSE_CATEGORIES,
    NAME,
    NAME_COLUMNS,
    STUDENT_HOUSES,
    STUDENT_TABLE,
    sorting_hat_bad_mapping,
    sorting_hat_serial,
    sorting_hat_vectorized,
)
from vivarium.framework.results.manager import ResultsManager
from vivarium.framework.results.stratification import (
    Stratification,
    get_mapped_col_name,
    get_original_col_name,
)


@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (  # expected output for vectorized
            sorting_hat_vectorized,
            True,
        ),
        (  # expected output for non-vectorized
            sorting_hat_serial,
            False,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_stratification(
    mapper: Callable[[pd.DataFrame], pd.Series[str]] | Callable[[pd.Series[str]], str],
    is_vectorized: bool,
) -> None:
    my_stratification = Stratification(
        name=NAME,
        sources=NAME_COLUMNS,
        categories=HOUSE_CATEGORIES,
        excluded_categories=[],
        mapper=mapper,
        is_vectorized=is_vectorized,
    )
    output = my_stratification.stratify(STUDENT_TABLE)
    assert isinstance(output, pd.Series)
    assert output.eq(STUDENT_HOUSES).all()


@pytest.mark.parametrize(
    "sources, categories, mapper, msg_match",
    [
        (
            [],
            HOUSE_CATEGORIES,
            None,
            (
                f"No mapper but 0 stratification sources are provided for stratification {NAME}. "
                "The list of sources must be of length 1 if no mapper is provided."
            ),
        ),
        (
            NAME_COLUMNS,
            HOUSE_CATEGORIES,
            None,
            (
                f"No mapper but {len(NAME_COLUMNS)} stratification sources are provided for stratification {NAME}. "
                "The list of sources must be of length 1 if no mapper is provided."
            ),
        ),
        (
            [],
            HOUSE_CATEGORIES,
            sorting_hat_vectorized,
            "The sources argument must be non-empty.",
        ),
        (
            NAME_COLUMNS,
            [],
            FileNotFoundError,
            "The categories argument must be non-empty.",
        ),
    ],
    ids=[
        "no_mapper_empty_sources",
        "no_mapper_multiple_sources",
        "with_mapper_empty_sources",
        "empty_categories",
    ],
)
def test_stratification_init_raises(
    sources: list[str],
    categories: list[str],
    mapper: Callable[[pd.DataFrame], pd.Series[str]] | Callable[[pd.Series[str]], str],
    msg_match: str,
) -> None:
    with pytest.raises(ValueError, match=re.escape(msg_match)):
        Stratification(NAME, sources, categories, [], mapper, True)


@pytest.mark.parametrize(
    "sources, mapper, is_vectorized, expected_exception, error_match",
    [
        (
            NAME_COLUMNS,
            sorting_hat_bad_mapping,
            False,
            ValueError,
            "Invalid values mapped to hogwarts_house: {'pancakes'}",
        ),
        (
            ["middle_initial"],
            sorting_hat_vectorized,
            True,
            KeyError,
            "None of [Index(['middle_initial'], dtype='object')] are in the [columns]",
        ),
        (
            NAME_COLUMNS,
            sorting_hat_serial,
            True,
            Exception,  # Can be any exception
            "",  # Can be any error message
        ),
        (
            NAME_COLUMNS,
            sorting_hat_vectorized,
            False,
            Exception,  # Can be any exception
            "",  # Can be any error message
        ),
        (
            NAME_COLUMNS,
            lambda df: pd.Series(np.nan, index=df.index),
            True,
            ValueError,
            f"Invalid values mapped to hogwarts_house: {{{np.nan}}}",
        ),
    ],
    ids=[
        "unknown_category",
        "source_not_in_population_columns",
        "vectorized_with_serial_mapper",
        "not_vectorized_with_vectorized_mapper",
        "mapper_returns_null",
    ],
)
def test_stratification_call_raises(
    sources: list[str],
    mapper: Callable[[pd.DataFrame], pd.Series[str]] | Callable[[pd.Series[str]], str],
    is_vectorized: bool,
    expected_exception: type[Exception],
    error_match: str,
) -> None:
    my_stratification = Stratification(
        NAME, sources, HOUSE_CATEGORIES, [], mapper, is_vectorized
    )
    with pytest.raises(expected_exception, match=re.escape(error_match)):
        my_stratification.stratify(STUDENT_TABLE)


@pytest.mark.parametrize("default_stratifications", [["age", "sex"], ["age"], []])
def test_setting_default_stratifications(
    default_stratifications: list[str], mocker: MockerFixture
) -> None:
    """Test that default stratifications are set as expected."""
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification.default = default_stratifications

    mgr.setup(builder)

    assert mgr._results_context.default_stratifications == default_stratifications


def test_get_mapped_column_name() -> None:
    assert get_mapped_col_name("foo") == "foo_mapped_values"


@pytest.mark.parametrize(
    "col_name, expected",
    [
        ("foo_mapped_values", "foo"),
        ("foo", "foo"),
        ("foo_mapped_values_mapped_values", "foo_mapped_values"),
        ("foo_mapped_values2", "foo_mapped_values2"),
        ("_mapped_values_foo", "_mapped_values_foo"),
        ("_mapped_values_foo_mapped_values", "_mapped_values_foo"),
    ],
)
def test_get_original_col_name(col_name: str, expected: str) -> None:
    assert get_original_col_name(col_name) == expected
