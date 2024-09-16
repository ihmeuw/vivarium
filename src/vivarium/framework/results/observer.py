# mypy: ignore-errors
"""
=========
Observers
=========

An observer is a component that is responsible for registering
:class:`observations <vivarium.framework.results.observation.BaseObservation>`
to the simulation.

The provided :class:`Observer` class is an abstract base class that should be subclassed
by concrete observers. Each concrete observer is required to implement a
`register_observations` method that registers all required observations.

"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, List

import pandas as pd
import scipy.sparse as sparse

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.event import Event


class Observer(Component, ABC):
    """An abstract base class intended to be subclassed by observer components.

    Notes
    -----
        A `register_observation` method must be defined in the subclass.

    """

    def __init__(self) -> None:
        super().__init__()
        self.results_dir = None

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                },
            },
        }

    def get_configuration_name(self) -> str:
        """Return the name of a concrete observer for use in the configuration"""
        return self.name.split("_observer")[0]

    @abstractmethod
    def register_observations(self, builder: Builder) -> None:
        """(Required). Register observations with within each observer."""
        pass

    def setup_component(self, builder: Builder) -> None:
        """Set up the observer component."""
        super().setup_component(builder)
        self.register_observations(builder)
        self.set_results_dir(builder)

    def set_results_dir(self, builder: Builder) -> None:
        """Define the results directory from the configuration."""
        self.results_dir = (
            builder.configuration.to_dict()
            .get("output_data", {})
            .get("results_directory", None)
        )

class FullHistoryObserver(Component):

    # other options:
    # - use scipy.sparse.csc_array:
    #   - map non-numeric columns to numeric
    #     - map categorical to integer
    #   - save mapping strategy so that it can be inverted
    #   - save column names
    #   - convert to csc_array

    @property
    def columns_required(self) -> Optional[List[str]]:
        return []

    def __init__(self):
        super().__init__()
        self.time_step_counter = 0.0
        self.clock = None
        self.initial_state = pd.DataFrame()
        self.recorded_state = pd.DataFrame()
        self.changes = []
        # self.column_maps = {
        #     column_name: self._create_map(col_map)
        #     for column_name, col_map in [
        #         ("tracked", {"tracked": 1, "untracked": 2}),
        #         ("sex", {"Male": 1, "Female": 2}),
        #         ("alive", {"alive": 1, "dead": 2}),
        #         (
        #             "lower_respiratory_infections",
        #             {
        #                 "susceptible_to_lower_respiratory_infections": 1,
        #                 "lower_respiratory_infections": 2,
        #             }
        #         ),
        #         # ("entrance_time", {"datetime": None}),
        #     ]
        # }

    # @staticmethod
    # def _create_map(col_map: dict[Any, Number]) -> pd.DataFrame:
    #     return pd.DataFrame(
    #         {
    #             "mapped_value": list(col_map.values()),
    #             "true_value": list(col_map.keys()),
    #         }
    #     )

    def setup(self, builder: Builder) -> None:
        self.clock = builder.time.clock()

    def on_time_step_prepare(self, event: Event) -> None:
        if self.initial_state.empty:
            self.initial_state = self.population_view.get(event.index)
            self.recorded_state = self.initial_state.drop(columns="age")


    def on_collect_metrics(self, event: Event) -> None:
        current_state = self.population_view.get(event.index).drop(columns="age")
        # todo deal with new rows
        changed_elements = current_state != self.recorded_state
        self.recorded_state = current_state.copy()
        for column in current_state.columns:
            unchanged = ~changed_elements[column]
            # fixme this will have issue for dtypes that have no nan value
            # fixme this will have issue if 0.0 is a valid value for the column
            #  can address this by converting all 0.0 to nan first
            current_state.loc[unchanged, column] = 0

            # if column in self.column_maps:
            #     mapper = self.column_maps[column].set_index("true_value").squeeze()
            #     current_state.loc[~unchanged, column] = current_state.loc[~unchanged, column].map(mapper)
            # else:
            #     # todo fix this
            #     current_state.loc[~unchanged, column] = 1.0
            current_state.loc[~unchanged, column] = 12345

        current_state = current_state.astype(float)
        sparse_diffs = sparse.csc_array(current_state)
        self.changes.append(sparse_diffs)
        self.time_step_counter += 1


    def on_simulation_end(self, event: Event) -> None:
        root_dir = Path("/home/rmudambi/scratch/test-sim")
        c = sparse.hstack(self.changes)
        sparse.save_npz(root_dir / "changes.npz", c)
