"""This module contains various helper classes used for testing since
classes cannot be pytest fixtures.
"""

import pandas as pd

from vivarium.framework.components.manager import Component
from vivarium.framework.engine import Builder


class CatToyObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(name="cat_toy")


class CatActivityObserver(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_observation(
            name="cat_activity",
            additional_stratifications=["favorite_activity"],
            excluded_stratifications=["favorite_toy"],
        )


class CatResultsStratifier(Component):
    def setup(self, builder: Builder) -> None:
        builder.results.register_stratification(
            "personality", ["psycopath", "cantankerous"], requires_columns=["foo"]
        )
        builder.results.register_stratification(
            "favorite_toy", ["string", "human_face"], requires_columns=["foo"]
        )
        builder.results.register_stratification(
            "favorite_activity", ["sleep", "eat"], requires_columns=["foo"]
        )
