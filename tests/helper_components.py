"""This module contains various helper classes used for testing since
classes cannot be pytest fixtures.
"""

import pandas as pd

from vivarium.framework.components.manager import Component
from vivarium.framework.engine import Builder


class MeasureOneObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(name="measure_one")


class MeasureTwoObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(name="measure_two")


class YearSexResultsStratifier(Component):
    def setup(self, builder: Builder) -> None:
        self.start_year = 2024
        self.end_year = 2027

        builder.results.register_stratification(
            "current_year",
            [str(year) for year in range(self.start_year, self.end_year + 1)],
            self.map_year,
            is_vectorized=True,
            requires_columns=["current_time"],
        )
        builder.results.register_stratification(
            "sex", ["Female", "Male"], requires_columns=["sex"]
        )

    @staticmethod
    def map_year(pop: pd.DataFrame) -> pd.Series:
        return pop.squeeze(axis=1).dt.year.apply(str)
