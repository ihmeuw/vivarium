"""
====================
The Simulation Clock
====================

The components here provide implementations of different kinds of simulation
clocks for use in ``vivarium``.

For more information about time in the simulation, see the associated
:ref:`concept note <time_concept>`.

"""
from datetime import datetime, timedelta
from numbers import Number
from typing import TYPE_CHECKING, Callable, List, Union

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population.population_view import PopulationView
    from vivarium.framework.event import Event

from vivarium.framework.values import list_combiner
from vivarium.manager import Manager

Time = Union[pd.Timestamp, datetime, Number]
Timedelta = Union[pd.Timedelta, timedelta, Number]
NumberLike = Union[np.ndarray, pd.Series, pd.DataFrame, Number]


class SimulationClock(Manager):
    """A base clock that includes global clock and a pandas series of clocks for each simulant"""

    @property
    def name(self):
        return "simulation_clock"

    @property
    def columns_created(self) -> List[str]:
        return ["next_event_time", "step_size"]

    @property
    def time(self) -> Time:
        """The current simulation time."""
        if not self._clock_time:
            raise ValueError("No start time provided")
        return self._clock_time

    @property
    def stop_time(self) -> Time:
        """The time at which the simulation will stop."""
        if not self._stop_time:
            raise ValueError("No stop time provided")
        return self._stop_time

    @property
    def minimum_step_size(self) -> Timedelta:
        """The minimum step size."""
        if not self._minimum_step_size:
            raise ValueError("No minimum step size provided")
        return self._minimum_step_size

    @property
    def default_step_size(self) -> Timedelta:
        """The default varied step size."""
        if not self._default_step_size:
            raise ValueError("No default step size provided")
        return self._default_step_size

    @property
    def step_size(self) -> Timedelta:
        """The size of the next time step."""
        if not self._clock_step_size:
            raise ValueError("No step size provided")
        return self._clock_step_size

    @property
    def event_time(self) -> Time:
        "Convenience method for event time, or clock + step"
        return self.time + self.step_size

    def __init__(self):
        self._clock_time: Time = None
        self._stop_time: Time = None
        self._minimum_step_size: Timedelta = None
        self._default_step_size: Timedelta = None
        self._clock_step_size: Timedelta = None
        self.individual_clocks: PopulationView = None

    def setup(self, builder: "Builder"):
        self.step_size_pipeline = builder.value.register_value_producer(
            "simulant_step_size",
            source=lambda idx: [pd.Series(np.nan, index=idx).astype("timedelta64[ns]")],
            preferred_combiner=list_combiner,
            preferred_post_processor=self.step_size_post_processor,
        )
        builder.population.initializes_simulants(
            self.on_initialize_simulants, creates_columns=self.columns_created
        )
        builder.event.register_listener("post_setup", self.on_post_setup)
        self.individual_clocks = builder.population.get_view(columns=self.columns_created)

    def on_post_setup(self, event: "Event"):
        if not self.step_size_pipeline.mutators:
            ## No components modify the step size, so we use the default
            ## and remove the population view
            self.individual_clocks = None

    def on_initialize_simulants(self, pop_data):
        """Sets the next_event_time and step_size columns for each simulant"""
        if self.individual_clocks:
            clocks_to_initialize = pd.DataFrame(
                {
                    "next_event_time": [self.event_time] * len(pop_data.index),
                    "step_size": [self.step_size] * len(pop_data.index),
                },
                index=pop_data.index,
            )
            self.individual_clocks.update(clocks_to_initialize)

    def simulant_next_event_times(self, index: pd.Index) -> pd.Series:
        """The next time each simulant will be updated."""
        if not self.individual_clocks:
            return None
        return self.individual_clocks.subview(["next_event_time"]).get(index).squeeze(axis=1)

    def simulant_step_sizes(self, index: pd.Index) -> pd.Series:
        """The step size for each simulant."""
        if not self.individual_clocks:
            return None
        return self.individual_clocks.subview(["step_size"]).get(index).squeeze(axis=1)

    def step_backward(self) -> None:
        """Rewinds the clock by the current step size."""
        self._clock_time -= self.step_size

    def step_forward(self, index: pd.Index) -> None:
        """Advances the clock by the current step size, and updates aligned simulant clocks."""
        self._clock_time += self.step_size
        if self.individual_clocks:
            update_index = self.get_active_population(index, self.time)
            pop_to_update = self.individual_clocks.get(update_index)
            if not pop_to_update.empty:
                pop_to_update["step_size"] = self.step_size_pipeline(update_index)
                pop_to_update["next_event_time"] = self.time + pop_to_update["step_size"]
                self.individual_clocks.update(pop_to_update)
            self._clock_step_size = self.simulant_next_event_times(index).min() - self.time

    def get_active_population(self, index: pd.Index, time: Time) -> pd.Index:
        """Gets population that is aligned with global clock"""
        if not self.individual_clocks:
            return index
        next_event_times = self.simulant_next_event_times(index)
        return next_event_times[next_event_times <= time].index

    def step_size_post_processor(self, values: List[NumberLike], _) -> pd.Series:
        """Computes the largest feasible step size for each simulant. This is the smallest component-modified
        step size (rounded down to increments of the minimum step size), or the global step size, whichever is larger.
        If no components modify the step size, we default to the global step size.

        Parameters
        ----------
        values
            A list of step sizes

        Returns
        -------
        pandas.Series
            The largest feasible step size for each simulant


        """

        min_modified = pd.DataFrame(values).min(axis=0).fillna(self.default_step_size)
        ## Rescale pipeline values to global minimum step size
        discretized_step_sizes = (
            np.floor(min_modified / self.minimum_step_size).replace(0, 1)
            * self.minimum_step_size
        )
        ## Make sure we don't get zero
        return discretized_step_sizes


class SimpleClock(SimulationClock):
    """A unitless step-count based simulation clock."""

    CONFIGURATION_DEFAULTS = {
        "time": {
            "start": 0,
            "end": 100,
            "step_size": 1,
        }
    }

    @property
    def name(self):
        return "simple_clock"

    def setup(self, builder):
        super().setup(builder)
        time = builder.configuration.time
        self._clock_time = time.start
        self._stop_time = time.end
        self._minimum_step_size = time.step_size
        self._default_step_size = (
            time.default_step_size if time.default_step_size else self._minimum_step_size
        )
        self._clock_step_size = self._default_step_size

    def __repr__(self):
        return "SimpleClock()"


def get_time_stamp(time):
    return pd.Timestamp(time["year"], time["month"], time["day"])


class DateTimeClock(SimulationClock):
    """A date-time based simulation clock."""

    CONFIGURATION_DEFAULTS = {
        "time": {
            "start": {"year": 2005, "month": 7, "day": 2},
            "end": {
                "year": 2010,
                "month": 7,
                "day": 2,
            },
            "step_size": 1,
            "default_step_size": None,  # Days
        }
    }

    @property
    def name(self):
        return "datetime_clock"

    def setup(self, builder):
        super().setup(builder)
        time = builder.configuration.time
        self._clock_time = get_time_stamp(time.start)
        self._stop_time = get_time_stamp(time.end)
        self._minimum_step_size = pd.Timedelta(
            days=time.step_size // 1, hours=(time.step_size % 1) * 24
        )
        self._default_step_size = (
            time.default_step_size if time.default_step_size else self._minimum_step_size
        )
        self._clock_step_size = self._default_step_size

    def __repr__(self):
        return "DateTimeClock()"


class TimeInterface:
    def __init__(self, manager: SimulationClock):
        self._manager = manager

    def clock(self) -> Callable[[], Time]:
        """Gets a callable that returns the current simulation time."""
        return lambda: self._manager.time

    def step_size(self) -> Callable[[], Timedelta]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.step_size

    def simulant_next_event_times(self) -> Callable[[pd.Index], pd.Series]:
        """Gets a callable that returns the next event times for simulants."""
        return self._manager.simulant_next_event_times

    def simulant_step_sizes(self) -> Callable[[pd.Index], pd.Series]:
        """Gets a callable that returns the simulant step sizes."""
        return self._manager.simulant_step_sizes
