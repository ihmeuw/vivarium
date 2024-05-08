from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.results import METRICS_COLUMN


class Observer(Component, ABC):
    """An abstract base class intended to be subclassed by observer components.
    The primary purpose of this class is to provide attributes required by
    the subclass `report` method.

    Note that a `register_observation` method must be defined in the subclass.
    """

    def __init__(self):
        super().__init__()
        self.results_dir = None
        self.input_draw = None
        self.random_seed = None

    @abstractmethod
    def register_observations(self, builder: Builder) -> None:
        """(Required). Register observations with within each observer."""
        pass

    def setup_component(self, builder: Builder) -> None:
        super().setup_component(builder)
        self.register_observations(builder)
        self.get_report_attributes(builder)

    def get_report_attributes(self, builder: Builder) -> None:
        """Define commonly-used attributes for reporting."""
        self.results_dir = (
            builder.configuration.to_dict()
            .get("output_data", {})
            .get("results_directory", None)
        )
        self.input_draw = (
            builder.configuration.to_dict()
            .get("input_data", {})
            .get("input_draw_number", None)
        )
        self.random_seed = (
            builder.configuration.to_dict().get("randomness", {}).get("random_seed", None)
        )

    ##################
    # Report methods #
    ##################

    def dataframe_to_csv(self, measure: str, results: pd.DataFrame) -> None:
        # Add extra cols
        results["measure"] = measure
        results["random_seed"] = self.random_seed
        results["input_draw"] = self.input_draw
        # Sort the columns such that the stratifications (index) are first
        # and METRICS_COLUMN is last and sort the rows by the stratifications.
        other_cols = [c for c in results.columns if c != METRICS_COLUMN]
        results = results[other_cols + [METRICS_COLUMN]].sort_index().reset_index()
        results.to_csv(Path(self.results_dir) / f"{measure}.csv", index=False)


class StratifiedObserver(Observer):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.name.split("_observer")[0]: {
                    "exclude": [],
                    "include": [],
                },
            },
        }
