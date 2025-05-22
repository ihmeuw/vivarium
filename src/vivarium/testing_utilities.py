"""
==========================
Vivarium Testing Utilities
==========================

Utility functions and classes to make testing ``vivarium`` components easier.

"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.randomness.index_map import IndexMap
from vivarium.framework.randomness.stream import RandomnessStream
from vivarium.types import ClockStepSize, ClockTime


class NonCRNTestPopulation(Component):
    CONFIGURATION_DEFAULTS = {
        "population": {
            "initialization_age_min": 0,
            "initialization_age_max": 100,
            "untracking_age": None,
        },
    }

    @property
    def columns_created(self) -> list[str]:
        return ["age", "sex", "location", "alive", "entrance_time", "exit_time"]

    def setup(self, builder: Builder) -> None:
        self.config = builder.configuration
        self.randomness = builder.randomness.get_stream(
            "population_age_fuzz", initializes_crn_attributes=True
        )

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        age_start = pop_data.user_data.get(
            "age_start", self.config.population.initialization_age_min
        )
        age_end = pop_data.user_data.get(
            "age_end", self.config.population.initialization_age_max
        )
        location = self.config.input_data.location

        population = _non_crn_build_population(
            pop_data.index,
            age_start,
            age_end,
            location,
            pop_data.creation_time,
            pop_data.creation_window,
            self.randomness,
        )
        self.population_view.update(population)

    def on_time_step(self, event: Event) -> None:
        population = self.population_view.get(event.index, query="alive == 'alive'")
        # This component won't work if event.step_size is an int
        if not isinstance(event.step_size, int):
            population["age"] += event.step_size / pd.Timedelta(days=365)
        self.population_view.update(population)


class TestPopulation(NonCRNTestPopulation):
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        self.age_randomness = builder.randomness.get_stream(
            "age_initialization", initializes_crn_attributes=True
        )
        self.register = builder.randomness.register_simulants

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        age_start = pop_data.user_data.get(
            "age_start", self.config.population.initialization_age_min
        )
        age_end = pop_data.user_data.get(
            "age_end", self.config.population.initialization_age_max
        )
        age_draw = self.age_randomness.get_draw(pop_data.index)
        if age_start == age_end:
            # This component won't work if creation window is an int
            if not isinstance(pop_data.creation_window, int):
                age = (
                    age_draw * (pop_data.creation_window / pd.Timedelta(days=365)) + age_start
                )
        else:
            age = age_draw * (age_end - age_start) + age_start

        core_population = pd.DataFrame(
            {"entrance_time": pop_data.creation_time, "age": age.values}, index=pop_data.index
        )
        self.register(core_population)

        if "location" in self.config.input_data.keys():
            location = self.config.input_data.location
        else:
            location = self.randomness.choice(
                pop_data.index, ["USA", "Canada", "Mexico"], additional_key="location_choice"
            )
        population = _build_population(core_population, location, self.randomness)
        self.population_view.update(population)


def _build_population(
    core_population: pd.DataFrame, location: str, randomness_stream: RandomnessStream
) -> pd.DataFrame:
    index = core_population.index

    population = pd.DataFrame(
        {
            "age": core_population["age"],
            "entrance_time": core_population["entrance_time"],
            "sex": randomness_stream.choice(
                index, ["Male", "Female"], additional_key="sex_choice"
            ),
            "alive": pd.Series("alive", index=index),
            "location": location,
            "exit_time": pd.NaT,
        },
        index=index,
    )
    return population


def _non_crn_build_population(
    index: pd.Index[int],
    age_start: float,
    age_end: float,
    location: str,
    creation_time: ClockTime,
    creation_window: ClockStepSize,
    randomness_stream: RandomnessStream,
) -> pd.DataFrame:
    if age_start == age_end:
        if not isinstance(creation_window, int):
            age = (
                randomness_stream.get_draw(index) * (creation_window / pd.Timedelta(days=365))
                + age_start
            )
    else:
        age = randomness_stream.get_draw(index) * (age_end - age_start) + age_start

    population = pd.DataFrame(
        {
            "age": age,
            "sex": randomness_stream.choice(
                index, ["Male", "Female"], additional_key="sex_choice"
            ),
            "alive": pd.Series("alive", index=index),
            "location": location,
            "entrance_time": creation_time,
            "exit_time": pd.NaT,
        },
        index=index,
    )
    return population


def build_table(
    value: Any,
    parameter_columns: dict[str, Sequence[int]] = {
        "age": (0, 125),
        "year": (1990, 2020),
    },
    key_columns: dict[str, Sequence[Any]] = {"sex": ("Female", "Male")},
    value_columns: list[str] = ["value"],
) -> pd.DataFrame:
    """

    Parameters
    ----------
    value
        Value(s) to put in the value columns of a lookup table.
    parameter_columns
        A dictionary where the keys are parameter (continuous) columns of a lookup table
        and the values are tuple of the range (inclusive) for that column.
    key_columns
        A dictionary where the keys are key (categorical) columns of a lookup table
        and the values are a tuple of the categories for that column
    value_columns
        A list of value columns that will appear in the returned lookup table

    Returns
    -------
        A pandas dataframe that has the cartesian product of the range of all parameter columns
        and the values of the key columns.
    """
    if not isinstance(value, list):
        value = [value] * len(value_columns)

    if len(value) != len(value_columns):
        raise ValueError("Number of values must match number of value columns")

    # Get product of parameter columns
    range_parameter_product = {
        key: list(range(value[0], value[1])) for key, value in parameter_columns.items()
    }
    # Build out dict of items we will need cartesian product of to make dataframe
    product_dict = dict(range_parameter_product)
    product_dict.update(key_columns)  # type: ignore [arg-type]
    products = product(*product_dict.values())

    rows = []
    for item in products:
        # Note: item is going to be a tuple of the cartesian product of the key column values and parameter column
        # values and will be ordered in the order of the parameter then key dict keys
        r_values = []
        for val in value:
            if val is None:
                r_values.append(np.random.random())
            elif callable(val):
                r_values.append(val(item))
            else:
                r_values.append(val)

        # Get list of values for rows (index values)
        key_columns_index_values = list(item[len(parameter_columns) :])
        # Transform parameter column values
        parameter_columns_index_values = item[: len(parameter_columns)]
        # Create intervals for parameter columns. Example year, year+1 for year_start and year_end
        unpacked_parameter_columns_index_values: list[Any] = [
            v for val in parameter_columns_index_values for v in (val, val + 1)
        ]
        rows.append(
            unpacked_parameter_columns_index_values + key_columns_index_values + r_values
        )

    # Make list of parameter column names
    parameter_column_names = [
        col_name for col in parameter_columns for col_name in (f"{col}_start", f"{col}_end")
    ]

    return pd.DataFrame(
        rows,
        columns=parameter_column_names + list(key_columns.keys()) + value_columns,
    )


def get_randomness(
    key: str = "test",
    clock: Callable[[], pd.Timestamp | datetime | int] = lambda: pd.Timestamp(1990, 7, 2),
    seed: int = 12345,
    initializes_crn_attributes: bool = False,
) -> RandomnessStream:
    return RandomnessStream(
        key,
        clock,
        seed=seed,
        index_map=IndexMap(),
        initializes_crn_attributes=initializes_crn_attributes,
    )


def metadata(file_path: str, layer: str = "override") -> dict[str, str]:
    return {"layer": layer, "source": str(Path(file_path).resolve())}
