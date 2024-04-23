import itertools
from typing import List

import numpy as np
import pandas as pd

from vivarium.framework.components.manager import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData

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
RECORDS = list(itertools.product(CATEGORIES, FAMILIARS, POWER_LEVELS, TRACKED_STATUSES))
BASE_POPULATION = pd.DataFrame(data=RECORDS, columns=COL_NAMES)

CONFIG = {
    "stratification": {
        "default": ["student_house", "power_level"],
    },
}


##################
# Helper classes #
##################


class Hogwarts(Component):
    @property
    def columns_created(self) -> List[str]:
        return [
            "student_house",
            "familiar",
            "power_level",
            "house_points",
            "quidditch_wins",
        ]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        size = len(pop_data.index)
        rng = np.random.default_rng(42)
        initialization_data = pd.DataFrame(
            {
                "student_house": rng.choice(STUDENT_HOUSES, size=size),
                "familiar": rng.choice(FAMILIARS, size=size),
                "power_level": rng.choice([str(lvl) for lvl in POWER_LEVELS], size=size),
                "house_points": 0,
                "quidditch_wins": 0,
            },
            index=pop_data.index,
        )
        self.population_view.update(initialization_data)

    def on_time_step(self, pop_data: SimulantData) -> None:
        update = self.population_view.get(pop_data.index)
        update["house_points"] = 0
        update["quidditch_wins"] = 0
        # House points are stratified by 'student_house' and 'power_level'.
        # Let's have each wizard of gryffindor and of level 50 and 80 gain a point
        # on each time step.
        update.loc[
            (update["student_house"] == "gryffindor")
            & (update["power_level"].isin(["50", "80"])),
            "house_points",
        ] = 1
        # Quidditch wins are stratified by 'familiar' and 'power level'.
        # Let's have each wizard with a banana slug familiar gain a point
        # on each time step.
        update.loc[update["familiar"] == "banana_slug", "quidditch_wins"] = 1
        self.population_view.update(update)


class HousePointsObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="house_points",
            aggregator_sources=["house_points"],
            aggregator=sum,
            requires_columns=[
                "house_points",
                "student_house",
                "power_level",
            ],
        )


class FullyFilteredHousePointsObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="house_points",
            pop_filter="tracked==True & power_level=='one billion'",
            aggregator_sources=["house_points"],
            aggregator=sum,
            requires_columns=[
                "house_points",
                "student_house",
                "power_level",
            ],
        )


class QuidditchWinsObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="quidditch_wins",
            aggregator_sources=["quidditch_wins"],
            aggregator=sum,
            excluded_stratifications=["student_house"],
            additional_stratifications=["familiar"],
            requires_columns=[
                "quidditch_wins",
                "familiar",
                "power_level",
            ],
        )


class HogwartsResultsStratifier(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "student_house", list(STUDENT_HOUSES), requires_columns=["student_house"]
        )
        builder.results.register_stratification(
            "familiar", FAMILIARS, requires_columns=["familiar"]
        )
        builder.results.register_stratification(
            "power_level",
            [str(lvl) for lvl in POWER_LEVELS],
            requires_columns=["power_level"],
        )


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
