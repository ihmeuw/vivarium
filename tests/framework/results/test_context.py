import numpy as np
import pandas as pd
import pytest

from vivarium.framework.results.context import ResultsContext
from vivarium.framework.results.stratification import Stratification

from .mocks import (
    CATEGORIES,
    NAME,
    SOURCES,
    sorting_hat_vector,
    sorting_hat_serial,
    sorting_hat_bad_mapping,
)

@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized",
    [
        (  # expected Stratification for vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
        ),
        (  # expected Stratification for non-vectorized
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
        ),
    ],
)
def test_add_stratification(name, sources, categories, mapper, is_vectorized):
    ctx = ResultsContext()
    ctx.add_stratification(name, sources, categories, mapper, is_vectorized)
    expected_object = Stratification(name, sources, categories, mapper, is_vectorized)
    assert expected_object in ctx._stratifications


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_exception",
    [
        (  # map to a category that isn't in CATEGORIES
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_bad_mapping,
            False,
            TypeError,
        ),
        (  # sources not in population columns
            NAME,
            ["middle_initial"],
            CATEGORIES,
            sorting_hat_vector,
            True,
            TypeError,
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
def test_add_stratification_raises(
    name, sources, categories, mapper, is_vectorized, expected_exception
):
    ctx = ResultsContext()
    with pytest.raises(expected_exception):
        raise ctx.add_stratification(name, sources, categories, mapper, is_vectorized)
