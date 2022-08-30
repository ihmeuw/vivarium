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


def sorting_hat_vector(state_table: pd.DataFrame) -> pd.Series:
    return pd.Series(["gryffindor", "slytherin", "ravenclaw"])


def sorting_hat_serial(simulant_row: pd.Series) -> str:
    # assign everyone to hufflepuff
    return "hufflepuff"


@pytest.mark.parametrize(
    "name, sources, categories, mapper, is_vectorized, expected_output",
    [
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_vector,
            True,
            pd.Series(["gryffindor", "slytherin", "ravenclaw"]),
        ),
        (
            NAME,
            SOURCES,
            CATEGORIES,
            sorting_hat_serial,
            False,
            pd.Series(["hufflepuff", "hufflepuff", "hufflepuff"]),
        ),
    ],
)
def test_stratification(name, sources, categories, mapper, is_vectorized, expected_output):
    my_stratification = Stratification(name, sources, categories, mapper, is_vectorized)
    output = my_stratification(STUDENT_TABLE)[name]
    assert output.eq(expected_output).all()
