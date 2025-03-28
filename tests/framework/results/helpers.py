from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.results import VALUE_COLUMN
from vivarium.framework.results.observer import Observer
from vivarium.framework.results.stratification import Stratification
from vivarium.framework.values import Pipeline
from vivarium.types import ScalarMapper, VectorMapper

NAME = "hogwarts_house"
NAME_COLUMNS = ["first_name", "last_name"]
HOUSE_CATEGORIES = ["hufflepuff", "ravenclaw", "slytherin", "gryffindor"]
STUDENT_TABLE = pd.DataFrame(
    np.array([["harry", "potter"], ["severus", "snape"], ["luna", "lovegood"]]),
    columns=NAME_COLUMNS,
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
RECORDS = list(itertools.product(HOUSE_CATEGORIES, FAMILIARS, POWER_LEVELS, TRACKED_STATUSES))
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

RNG = np.random.default_rng(42)


##################
# Helper classes #
##################


class Hogwarts(Component):
    @property
    def columns_created(self) -> list[str]:
        return [
            "student_id",
            "student_house",
            "familiar",
            "power_level",
            "house_points",
            "quidditch_wins",
            "exam_score",
            "spell_power",
            "potion_power",
        ]

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        size = len(pop_data.index)
        initialization_data = pd.DataFrame(
            {
                "student_id": list(range(size)),
                "student_house": RNG.choice(STUDENT_HOUSES, size=size),
                "familiar": RNG.choice(FAMILIARS, size=size),
                "power_level": RNG.choice([lvl for lvl in POWER_LEVELS], size=size),
                "house_points": 0,
                "quidditch_wins": 0,
                "exam_score": 0.0,
            },
            index=pop_data.index,
        )
        # Assume power level components are evenly split
        initialization_data["spell_power"] = initialization_data["power_level"] / 2
        initialization_data["potion_power"] = initialization_data["power_level"] / 2
        self.population_view.update(initialization_data)

    def on_time_step(self, pop_data: Event) -> None:
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
        # Update everyones test score to increase by 10 points per time step
        update["exam_score"] += 10.0

        self.population_view.update(update)


class HousePointsObserver(Observer):
    """Observer that is stratified by multiple columns (the defaults,
    'student_house' and 'power_level_group')
    """

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="house_points",
            aggregator_sources=["house_points"],
            aggregator=lambda df: df.sum(),
            requires_columns=[
                "house_points",
            ],
            results_formatter=results_formatter,
        )


class FullyFilteredHousePointsObserver(Observer):
    """Same as `HousePointsObserver but with a filter that leaves no simulants"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="house_points",
            pop_filter="tracked==True & power_level=='one billion'",
            aggregator_sources=["house_points"],
            aggregator=lambda df: df.sum(),
            requires_columns=[
                "house_points",
            ],
        )


class QuidditchWinsObserver(Observer):
    """Observer that is stratified by a single column ('familiar')"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="quidditch_wins",
            aggregator_sources=["quidditch_wins"],
            aggregator=lambda df: df.sum(),
            excluded_stratifications=["student_house", "power_level_group"],
            additional_stratifications=["familiar"],
            requires_columns=[
                "quidditch_wins",
            ],
            results_formatter=results_formatter,
        )


class NoStratificationsQuidditchWinsObserver(Observer):
    """Same as above but no stratifications at all"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="no_stratifications_quidditch_wins",
            aggregator_sources=["quidditch_wins"],
            aggregator=lambda df: df.sum(),
            excluded_stratifications=["student_house", "power_level_group"],
            requires_columns=[
                "quidditch_wins",
            ],
            results_formatter=results_formatter,
        )


class MagicalAttributesObserver(Observer):
    """Observer whose aggregator returns a pd.Series instead of a float (which in
    turn results in a dataframe with multiple columns instead of just one
    'value' column)
    """

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_adding_observation(
            name="magical_attributes",
            aggregator=self._get_magical_attributes,
            excluded_stratifications=["student_house"],
            results_formatter=lambda *_: pd.DataFrame(),
        )

    def _get_magical_attributes(self, _: pd.DataFrame) -> pd.Series[float]:
        """Increase each level by one per time step"""
        return pd.Series([1.0, 1.0], ["spell_power", "potion_power"])


class ExamScoreObserver(Observer):
    """Observer that is not stratified and exam scores"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_concatenating_observation(
            name="exam_score",
            requires_columns=["student_id", "student_house", "exam_score"],
        )


class CatBombObserver(Observer):
    """Observer that counts the number of feral cats per house"""

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_stratified_observation(
            name="cat_bomb",
            pop_filter="familiar=='cat' and tracked==True",
            requires_columns=["familiar"],
            results_updater=self.update_cats,
            excluded_stratifications=["power_level_group"],
            aggregator_sources=["student_house"],
            aggregator=len,
        )

    def update_cats(self, existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
        no_cats_mask = existing_df["value"] == 0
        updated_df = existing_df
        updated_df.loc[no_cats_mask, "value"] = new_df["value"]
        updated_df.loc[~no_cats_mask, "value"] *= new_df["value"]
        return updated_df


class ValedictorianObserver(Observer):
    """Observer that records the valedictorian at each time step. All students
    have the same exam scores and so the valedictorian is chosen randomly.
    """

    def __init__(self) -> None:
        super().__init__()
        self.valedictorians: list[int] = []

    def register_observations(self, builder: Builder) -> None:
        builder.results.register_unstratified_observation(
            name="valedictorian",
            requires_columns=["event_time", "student_id", "exam_score"],
            results_gatherer=self.choose_valedictorian,  # type: ignore [arg-type]
            results_updater=self.update_valedictorian,
        )

    def choose_valedictorian(self, df: pd.DataFrame) -> pd.DataFrame:
        eligible_students = df.loc[~df["student_id"].isin(self.valedictorians), "student_id"]
        valedictorian: int = RNG.choice(eligible_students)
        self.valedictorians.append(valedictorian)
        return df[df["student_id"] == valedictorian]

    def update_valedictorian(
        self, _existing_df: pd.DataFrame, new_df: pd.DataFrame
    ) -> pd.DataFrame:
        return new_df


class HogwartsResultsStratifier(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_stratification(
            name="student_house",
            categories=list(STUDENT_HOUSES),
            requires_columns=["student_house"],
        )
        builder.results.register_stratification(
            name="familiar", categories=FAMILIARS, requires_columns=["familiar"]
        )
        builder.results.register_binned_stratification(
            "power_level",
            "power_level_group",
            POWER_LEVEL_BIN_EDGES,
            POWER_LEVEL_GROUP_LABELS,
        )


##################
# Helper methods #
##################


def results_formatter(
    measure: str,
    results: pd.DataFrame,
) -> pd.DataFrame:
    """An test use case of an observer's report method that writes a DataFrame to a CSV file."""
    # Add extra cols
    results["measure"] = measure
    # Sort the columns such that the stratifications (index) are first
    # and VALUE_COLUMN is last and sort the rows by the stratifications.
    other_cols = [c for c in results.columns if c != VALUE_COLUMN]
    return results[other_cols + [VALUE_COLUMN]].sort_index().reset_index()


def sorting_hat_vectorized(state_table: pd.DataFrame) -> pd.Series[str]:
    sorted_series = state_table.apply(sorting_hat_serial, axis=1)
    return sorted_series


def sorting_hat_serial(simulant_row: pd.Series[str]) -> str:
    first_name = simulant_row[0]
    last_name = simulant_row[1]
    if first_name == "harry":
        return "gryffindor"
    if first_name == "luna":
        return "ravenclaw"
    if last_name == "snape":
        return "slytherin"
    return "hufflepuff"


def sorting_hat_bad_mapping(simulant_row: pd.Series[str]) -> str:
    # Return something not in CATEGORIES
    return "pancakes"


def verify_stratification_added(
    stratification_list: list[Stratification],
    name: str,
    sources: list[str],
    categories: list[str],
    excluded_categories: list[str],
    mapper: VectorMapper | ScalarMapper,
    is_vectorized: bool,
) -> bool:
    """Verify that a Stratification object is in `stratification_list`"""
    matching_stratification_found = False
    for stratification in stratification_list:  # noqa
        # big equality check
        if (
            stratification.name == name
            and sorted(stratification.categories)
            == sorted([cat for cat in categories if cat not in excluded_categories])
            and sorted(stratification.excluded_categories) == sorted(excluded_categories)
            and stratification.mapper == mapper
            and stratification.is_vectorized == is_vectorized
            and sorted(stratification.sources) == sorted(sources)
        ):
            matching_stratification_found = True
            break
    return matching_stratification_found


# Mock for get_value call for Pipelines, returns a str instead of a Pipeline
def mock_get_value(self: Builder, name: str) -> Pipeline:
    if not isinstance(name, str):
        raise TypeError("Passed a non-string type to mock get_value(), check your pipelines.")
    return Pipeline(name)
