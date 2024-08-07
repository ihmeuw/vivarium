import numpy as np
import pandas as pd
import pytest

from tests.framework.results.helpers import (
    CATEGORIES,
    NAME,
    SOURCES,
    STUDENT_HOUSES,
    STUDENT_TABLE,
    sorting_hat_bad_mapping,
    sorting_hat_serial,
    sorting_hat_vector,
)
from vivarium.framework.results.manager import ResultsManager
from vivarium.framework.results.stratification import Stratification


#########
# Tests #
#########
@pytest.mark.parametrize(
    "mapper, is_vectorized",
    [
        (  # expected output for vectorized
            sorting_hat_vector,
            True,
        ),
        (  # expected output for non-vectorized
            sorting_hat_serial,
            False,
        ),
    ],
    ids=["vectorized_mapper", "non-vectorized_mapper"],
)
def test_stratification(mapper, is_vectorized):
    my_stratification = Stratification(NAME, SOURCES, CATEGORIES, [], mapper, is_vectorized)
    output = my_stratification(STUDENT_TABLE)[NAME]
    assert output.eq(STUDENT_HOUSES).all()


@pytest.mark.parametrize(
    "sources, categories",
    [
        (  # empty sources list with no defined mapper (default mapper)
            [],
            CATEGORIES,
        ),
        (  # sources list with more than one column with no defined mapper (default mapper)
            SOURCES,
            CATEGORIES,
        ),
        (  # empty sources list with no defined mapper (default mapper)
            [],
            CATEGORIES,
        ),
        (  # empty categories list
            SOURCES,
            [],
        ),
    ],
)
def test_stratification_init_raises(sources, categories):
    with pytest.raises(ValueError):
        assert Stratification(NAME, sources, categories, [], None, True)


@pytest.mark.parametrize(
    "sources, mapper, is_vectorized, expected_exception",
    [
        (
            SOURCES,
            sorting_hat_bad_mapping,
            False,
            ValueError,
        ),
        (
            ["middle_initial"],
            sorting_hat_vector,
            True,
            KeyError,
        ),
        (
            SOURCES,
            sorting_hat_serial,
            True,
            Exception,
        ),
        (
            SOURCES,
            sorting_hat_vector,
            False,
            Exception,
        ),
        (
            SOURCES,
            lambda df: pd.Series(np.nan, index=df.index),
            True,
            ValueError,
        ),
    ],
    ids=[
        "category_not_in_categories",
        "source_not_in_population_columns",
        "vectorized_with_serial_mapper",
        "not_vectorized_with_serial_mapper",
        "mapper_returns_null",
    ],
)
def test_stratification_call_raises(sources, mapper, is_vectorized, expected_exception):
    my_stratification = Stratification(NAME, sources, CATEGORIES, [], mapper, is_vectorized)
    with pytest.raises(expected_exception):
        raise my_stratification(STUDENT_TABLE)


@pytest.mark.parametrize("default_stratifications", [["age", "sex"], ["age"], []])
def test_setting_default_stratifications(default_stratifications, mocker):
    """Test that default stratifications are set as expected."""
    mgr = ResultsManager()
    builder = mocker.Mock()
    builder.configuration.stratification.default = default_stratifications

    mgr.setup(builder)

    assert mgr._results_context.default_stratifications == default_stratifications
