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
