import numpy as np
import pandas as pd
import pytest

from vivarium.framework.results.stratification import Stratification

##########################
# Mock data and fixtures #
##########################
NAME = "hogwarts_house"
SOURCES = ["first_name", "last_name"]
CATEGORIES = ["hufflepuff", "ravenclaw", "slytherin", "gryffindor"]
STUDENT_TABLE = pd.DataFrame(
    np.array([["harry", "potter"], ["severus", "snape"], ["luna", "lovegood"]]),
    columns=SOURCES,
)
STUDENT_HOUSES = pd.Series(["gryffindor", "slytherin", "ravenclaw"])


def sorting_hat_vector(state_table: pd.DataFrame) -> pd.Series:
    sorted_series = state_table.apply(sorting_hat_serial, axis=1)
    return sorted_series


def sorting_hat_serial(simulant_row: pd.Series) -> str:
    first_name = simulant_row[0]
    last_name = simulant_row[1]
    if first_name == "harry":
        return "gryffindor"
    if first_name == "luna":
        return "ravenclaw"
    if last_name == "snape":
        return "slytherin"
    return "hufflepuff"


def sorting_hat_bad_mapping(simulant_row: pd.Series) -> str:
    # Return something not in CATEGORIES
    return "pancakes"


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
