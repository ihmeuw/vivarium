import itertools
import math

import numpy as np
import pandas as pd

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

BIN_BINNED_COLUMN = "silly_bin"
BIN_SOURCE = "silly_level"
BIN_LABELS = ["not", "meh", "somewhat", "very", "extra"]
BIN_SILLY_BINS = [0, 20, 40, 60, 90]

COL_NAMES = ["house", "familiar", "power_level", "tracked"]
FAMILIARS = ["owl", "cat", "gecko", "banana_slug", "unladen_swallow"]
POWER_LEVELS = [i * 10 for i in range(5, 9)]
TRACKED_STATUSES = [True, False]
RECORDS = [
    (house, familiar, power_level, ts)
    for house, familiar, power_level, ts in itertools.product(
        CATEGORIES, FAMILIARS, POWER_LEVELS, TRACKED_STATUSES
    )
]
BASE_POPULATION = pd.DataFrame(data=RECORDS, columns=COL_NAMES)


##################
# Helper methods #
##################
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


def verify_stratification_added(
    stratification_list, name, sources, categories, mapper, is_vectorized
):
    """Verify that a :class: `vivarium.framework.results.stratification.Stratification` is in `stratification_list`"""
    matching_stratification_found = False
    for stratification in stratification_list:  # noqa
        # big equality check
        if (
            stratification.name == name
            and sorted(stratification.categories) == sorted(categories)
            and stratification.mapper == mapper
            and stratification.is_vectorized == is_vectorized
            and sorted(stratification.sources) == sorted(sources)
        ):
            matching_stratification_found = True
            break
    return matching_stratification_found


# Mock for get_value call for Pipelines, returns a str instead of a Pipeline
def mock_get_value(self, name: str):
    if not isinstance(name, str):
        raise TypeError("Passed a non-string type to mock get_value(), check your pipelines.")
    return name
