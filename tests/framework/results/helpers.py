import itertools
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from vivarium.framework.components.manager import Component
from vivarium.framework.engine import Builder
from vivarium.framework.population import SimulantData
from vivarium.framework.results import METRICS_COLUMN
from vivarium.framework.results.observer import StratifiedObserver

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
BIN_LABELS = ["meh", "somewhat", "very", "extra"]
BIN_SILLY_BIN_EDGES = [0, 20, 40, 60, 90]

COL_NAMES = ["house", "familiar", "power_level", "tracked"]
FAMILIARS = ["owl", "cat", "gecko", "banana_slug", "unladen_swallow"]
POWER_LEVELS = [20, 40, 60, 80]
POWER_LEVEL_BIN_EDGES = [0, 25, 50, 75, 100]
POWER_LEVEL_GROUP_LABELS = ["low", "medium", "high", "very high"]
TRACKED_STATUSES = [True, False]
RECORDS = list(itertools.product(CATEGORIES, FAMILIARS, POWER_LEVELS, TRACKED_STATUSES))
BASE_POPULATION = pd.DataFrame(data=RECORDS, columns=COL_NAMES)

HARRY_POTTER_CONFIG = {
    "time": {
        "start": {"year": 2024, "month": 4, "day": 22},
        "end": {"year": 2029, "month": 4, "day": 22},
        "step_size": 365,  # Days
    },
    "stratification": {
        "default": ["student_house", "power_level_group"],
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
                "power_level": rng.choice([lvl for lvl in POWER_LEVELS], size=size),
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
        # House points are stratified by 'student_house' and 'power_level_group'.
        # Let's have each wizard of gryffindor and of power level 20 or 80
        # gain a point on each time step.
        update.loc[
            (update["student_house"] == "gryffindor")
            & (update["power_level"].isin([20, 80])),
            "house_points",
        ] = 1
        # Quidditch wins are stratified by 'familiar'.
        # Let's have each wizard with a banana slug familiar gain a point
        # on each time step.
        update.loc[update["familiar"] == "banana_slug", "quidditch_wins"] = 1
        self.population_view.update(update)


class HousePointsObserver(StratifiedObserver):
    """Observer that is stratified by multiple columns (the defaults,
    'student_house' and 'power_level_group')
    """

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="house_points",
            aggregator_sources=["house_points"],
            aggregator=sum,
            requires_columns=[
                "house_points",
            ],
            report=self.write_results,
        )

    def write_results(self, measure: str, results: pd.DataFrame) -> None:
        dataframe_to_csv(
            measure, results, Path(self.results_dir), self.random_seed, self.input_draw
        )


class FullyFilteredHousePointsObserver(Component):
    """Same as `HousePointsObserver but with a filter that leaves no simulants"""

    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="house_points",
            pop_filter="tracked==True & power_level=='one billion'",
            aggregator_sources=["house_points"],
            aggregator=sum,
            requires_columns=[
                "house_points",
            ],
        )


class QuidditchWinsObserver(StratifiedObserver):
    """Observer that is stratified by a single column ('familiar')"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="quidditch_wins",
            aggregator_sources=["quidditch_wins"],
            aggregator=sum,
            excluded_stratifications=["student_house", "power_level_group"],
            additional_stratifications=["familiar"],
            requires_columns=[
                "quidditch_wins",
            ],
            report=self.write_results,
        )

    def write_results(self, measure: str, results: pd.DataFrame) -> None:
        dataframe_to_csv(
            measure, results, Path(self.results_dir), self.random_seed, self.input_draw
        )


class NoStratificationsQuidditchWinsObserver(StratifiedObserver):
    """Same as above but no stratifications at all"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="no_stratifications_quidditch_wins",
            aggregator_sources=["quidditch_wins"],
            aggregator=sum,
            excluded_stratifications=["student_house", "power_level_group"],
            requires_columns=[
                "quidditch_wins",
            ],
            report=self.write_results,
        )

    def write_results(self, measure: str, results: pd.DataFrame) -> None:
        dataframe_to_csv(
            measure, results, Path(self.results_dir), self.random_seed, self.input_draw
        )


class HogwartsResultsStratifier(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "student_house", list(STUDENT_HOUSES), requires_columns=["student_house"]
        )
        builder.results.register_stratification(
            "familiar", FAMILIARS, requires_columns=["familiar"]
        )
        builder.results.register_binned_stratification(
            "power_level",
            "power_level_group",
            POWER_LEVEL_BIN_EDGES,
            POWER_LEVEL_GROUP_LABELS,
            "column",
        )


##################
# Helper methods #
##################


def dataframe_to_csv(
    measure: str,
    results: pd.DataFrame,
    results_dir: Path,
    random_seed: str,
    input_draw: str,
) -> None:
    """An test use case of an observer's report method that writes a DataFrame to a CSV file."""
    # Add extra cols
    results["measure"] = measure
    results["random_seed"] = random_seed
    results["input_draw"] = input_draw
    # Sort the columns such that the stratifications (index) are first
    # and METRICS_COLUMN is last and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != METRICS_COLUMN]
    results = results[other_cols + [METRICS_COLUMN]].sort_index().reset_index()
    results.to_csv(results_dir / f"{measure}.csv", index=False)


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
