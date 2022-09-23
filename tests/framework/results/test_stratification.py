import numpy as np
import pandas as pd
import pytest

from vivarium.framework.results.stratification import Stratification

from .mocks import (
    CATEGORIES,
    NAME,
    SOURCES,
    STUDENT_HOUSES,
    STUDENT_TABLE,
    sorting_hat_vector,
    sorting_hat_serial,
    sorting_hat_bad_mapping,
)


#########
# Tests #
#########
@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_output",
    [
        (  # expected output for vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
            STUDENT_HOUSES,
        ),
        (  # expected output for non-vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
            STUDENT_HOUSES,
        ),
    ],
)
def test_stratification(name, sources, categories, mapper, is_vectorized, expected_output):
    my_stratification = Stratification(name, sources, categories, mapper, is_vectorized)
    output = my_stratification(STUDENT_TABLE)[name]
    assert output.eq(expected_output).all()


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_exception",
    [
        (  # empty sources list with no defined mapper (default mapper)
            NAME,
            [],
            CATEGORIES,
            None,
            True,
            ValueError,
        ),
        (  # sources list with more than one column with no defined mapper (default mapper)
            NAME,
            SOURCES,
            CATEGORIES,
            None,
            True,
            ValueError,
        ),
        (  # empty sources list with no defined mapper (default mapper)
            NAME,
            [],
            CATEGORIES,
            None,
            True,
            ValueError,
        ),
        (  # empty categories list
            NAME,
            SOURCES,
            [],
            None,
            True,
            ValueError,
        ),
    ],
)
def test_stratification_init_raises(
    name, sources, categories, mapper, is_vectorized, expected_exception
):
    with pytest.raises(expected_exception):
        assert Stratification(name, sources, categories, mapper, is_vectorized)


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_exception",
    [
        (  # map to a category that isn't in CATEGORIES
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_bad_mapping,
            False,
            ValueError,
        ),
        (  # sources not in population columns
            NAME,
            ["middle_initial"],
            CATEGORIES,
            sorting_hat_vector,
            True,
            KeyError,
        ),
        (  # is_vectorized=True with non-vectorized mapper
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            True,
            Exception,
        ),
        (  # is_vectorized=False with vectorized mapper
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            False,
            Exception,
        ),
    ],
)
def test_stratification_call_raises(
    name, sources, categories, mapper, is_vectorized, expected_exception
):
    my_stratification = Stratification(name, sources, categories, mapper, is_vectorized)
    with pytest.raises(expected_exception):
        raise my_stratification(STUDENT_TABLE)
