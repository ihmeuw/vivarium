"""
====================
The Simulation Clock
====================

The components here provide implementations of different kinds of simulation
clocks for use in ``vivarium``.

For more information about time in the simulation, see the associated
:ref:`concept note <time_concept>`.

"""

from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from vivarium.types import ClockStepSize, ClockTime

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population.population_view import PopulationView
    from vivarium.framework.event import Event
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import ValuesManager

from vivarium.framework.values import list_combiner
from vivarium.manager import Interface, Manager


class SimulationClock(Manager):
    """A base clock that includes global clock and a pandas series of clocks for each simulant"""

    @property
    def name(self) -> str:
        return "simulation_clock"

    @property
    def columns_created(self) -> list[str]:
        return ["next_event_time", "step_size"]

    @property
    def columns_required(self) -> list[str]:
        return ["tracked"]

    @property
    def time(self) -> ClockTime:
        """The current simulation time."""
        if self._clock_time is None:
            raise ValueError("No start time provided")
        return self._clock_time

    @property
    def stop_time(self) -> ClockTime:
        """The time at which the simulation will stop."""
        if self._stop_time is None:
            raise ValueError("No stop time provided")
        return self._stop_time

    @property
    def minimum_step_size(self) -> ClockStepSize:
        """The minimum step size."""
        if self._minimum_step_size is None:
            raise ValueError("No minimum step size provided")
        return self._minimum_step_size

    @property
    def standard_step_size(self) -> ClockStepSize:
        """The standard varied step size."""
        if self._standard_step_size is None:
            raise ValueError("No standard step size provided")
        elif self._clock_step_size == 0:
            raise ValueError("Step size cannot be equal to zero")
        return self._standard_step_size

    @property
    def step_size(self) -> ClockStepSize:
        """The size of the next time step."""
        if self._clock_step_size is None:
            raise ValueError("No step size provided")
        elif self._clock_step_size == 0:
            raise ValueError("Step size cannot be equal to zero")
        return self._clock_step_size

    @property
    def event_time(self) -> ClockTime:
        "Convenience method for event time, or clock + step"
        return self.time + self.step_size  # type: ignore [operator]

    @property
    def time_steps_remaining(self) -> int:
        number_steps_remaining = (self.stop_time - self.time) / self.step_size  # type: ignore [operator]
        if not isinstance(number_steps_remaining, (float, int)):
            raise ValueError("Invalid type for number of steps remaining")
        return math.ceil(number_steps_remaining)

    def __init__(self) -> None:
        self._clock_time: ClockTime | None = None
        self._stop_time: ClockTime | None = None
        self._minimum_step_size: ClockStepSize | None = None
        self._standard_step_size: ClockStepSize | None = None
        self._clock_step_size: ClockStepSize | None = None
        self._individual_clocks: PopulationView | None = None
        self._pipeline_name = "simulant_step_size"
        # TODO: Delegate this functionality to "tracked" or similar when appropriate
        self._simulants_to_snooze = pd.Index([])

    def setup(self, builder: "Builder") -> None:
        self._step_size_pipeline = builder.value.register_value_producer(
            self._pipeline_name,
            source=lambda idx: [pd.Series(np.nan, index=idx).astype("timedelta64[ns]")],
            preferred_combiner=list_combiner,
            preferred_post_processor=self.step_size_post_processor,
        )
        self.register_step_modifier = partial(
            builder.value.register_value_modifier,
            self._pipeline_name,
            component=self,
        )
        builder.population.initializes_simulants(self, creates_columns=self.columns_created)
        builder.event.register_listener("post_setup", self.on_post_setup)
        self._individual_clocks = builder.population.get_view(
            columns=self.columns_created + self.columns_required
        )

    def on_post_setup(self, event: "Event") -> None:
        if not self._step_size_pipeline.mutators:
            ## No components modify the step size, so we use the default
            ## and remove the population view
            self._individual_clocks = None

    def on_initialize_simulants(self, pop_data: "SimulantData") -> None:
        """Sets the next_event_time and step_size columns for each simulant"""
        if self._individual_clocks:
            clocks_to_initialize = pd.DataFrame(
                {
                    "next_event_time": [self.event_time] * len(pop_data.index),
                    "step_size": [self.step_size] * len(pop_data.index),
                },
                index=pop_data.index,
            )
            self._individual_clocks.update(clocks_to_initialize)

    def simulant_next_event_times(self, index: pd.Index[int]) -> pd.Series[ClockTime]:
        """The next time each simulant will be updated."""
        if not self._individual_clocks:
            return pd.Series(self.event_time, index=index)
        return self._individual_clocks.subview(["next_event_time", "tracked"]).get(index)[
            "next_event_time"
        ]

    def simulant_step_sizes(self, index: pd.Index[int]) -> pd.Series[ClockStepSize]:
        """The step size for each simulant."""
        if not self._individual_clocks:
            return pd.Series(self.step_size, index=index)
        return self._individual_clocks.subview(["step_size", "tracked"]).get(index)[
            "step_size"
        ]

    def step_backward(self) -> None:
        """Rewinds the clock by the current step size."""
        if self._clock_time is None:
            raise ValueError("No start time provided")
        self._clock_time -= self.step_size  # type: ignore [operator]

    def step_forward(self, index: pd.Index[int]) -> None:
        """Advances the clock by the current step size, and updates aligned simulant clocks."""
        self._clock_time += self.step_size  # type: ignore [assignment, operator]
        if self._individual_clocks and not index.empty:
            update_index = self.get_active_simulants(index, self.time)
            clocks_to_update = self._individual_clocks.get(update_index)
            if not clocks_to_update.empty:
                clocks_to_update["step_size"] = self._step_size_pipeline(update_index)
                # Simulants that were flagged to get moved to the end should have a next event time
                # of stop time + 1 minimum timestep
                clocks_to_update.loc[self._simulants_to_snooze, "step_size"] = (
                    self.stop_time + self.minimum_step_size - self.time  # type: ignore [operator]
                )
                # TODO: Delegate this functionality to "tracked" or similar when appropriate
                self._simulants_to_snooze = pd.Index([])
                clocks_to_update["next_event_time"] = (
                    self.time + clocks_to_update["step_size"]
                )
                self._individual_clocks.update(clocks_to_update)
            self._clock_step_size = self.simulant_next_event_times(index).min() - self.time  # type: ignore [operator]

    def get_active_simulants(self, index: pd.Index[int], time: ClockTime) -> pd.Index[int]:
        """Gets population that is aligned with global clock"""
        if index.empty or not self._individual_clocks:
            return index
        next_event_times = self.simulant_next_event_times(index)
        return next_event_times[next_event_times <= time].index

    def move_simulants_to_end(self, index: pd.Index[int]) -> None:
        if self._individual_clocks and not index.empty:
            self._simulants_to_snooze = self._simulants_to_snooze.union(index)

    def step_size_post_processor(self, value: Any, manager: ValuesManager) -> Any:
        """Computes the largest feasible step size for each simulant.

        This is the smallest component-modified step size (rounded down to increments
        of the minimum step size), or the global step size, whichever is larger.
        If no components modify the step size, we default to the global step size.

        Parameters
        ----------
        values
            A list of step sizes

        Returns
        -------
            The largest feasible step size for each simulant
        """

        min_modified = pd.DataFrame(value).min(axis=0).fillna(self.standard_step_size)
        ## Rescale pipeline values to global minimum step size
        discretized_step_sizes = (
            np.floor(min_modified / self.minimum_step_size).replace(0, 1)  # type: ignore [attr-defined, operator]
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
            "standard_step_size": None,
        }
    }

    @property
    def name(self) -> str:
        return "simple_clock"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        time = builder.configuration.time
        self._clock_time = time.start
        self._stop_time = time.end
        self._minimum_step_size = time.step_size
        self._standard_step_size = (
            time.standard_step_size if time.standard_step_size else self._minimum_step_size
        )
        self._clock_step_size = self._standard_step_size

    def __repr__(self) -> str:
        return "SimpleClock()"


def get_time_stamp(time: dict[str, int]) -> pd.Timestamp:
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
            "step_size": 1,  # Days
            "standard_step_size": None,  # Days
        }
    }

    @property
    def name(self) -> str:
        return "datetime_clock"

    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        time = builder.configuration.time
        self._clock_time = get_time_stamp(time.start)
        self._stop_time = get_time_stamp(time.end)
        self._minimum_step_size = pd.Timedelta(
            days=time.step_size // 1, hours=(time.step_size % 1) * 24
        )
        self._standard_step_size = (
            pd.Timedelta(
                days=time.standard_step_size // 1, hours=(time.standard_step_size % 1) * 24
            )
            if time.standard_step_size
            else self._minimum_step_size
        )
        self._clock_step_size = self._minimum_step_size

    def __repr__(self) -> str:
        return "DateTimeClock()"


class TimeInterface(Interface):
    def __init__(self, manager: SimulationClock) -> None:
        self._manager = manager

    def clock(self) -> Callable[[], ClockTime]:
        """Gets a callable that returns the current simulation time."""
        return lambda: self._manager.time

    def step_size(self) -> Callable[[], ClockStepSize]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.step_size

    def simulant_next_event_times(self) -> Callable[[pd.Index[int]], pd.Series[ClockTime]]:
        """Gets a callable that returns the next event times for simulants."""
        return self._manager.simulant_next_event_times

    def simulant_step_sizes(self) -> Callable[[pd.Index[int]], pd.Series[ClockStepSize]]:
        """Gets a callable that returns the simulant step sizes."""
        return self._manager.simulant_step_sizes

    def move_simulants_to_end(self) -> Callable[[pd.Index[int]], None]:
        """Gets a callable that moves simulants to the end of the simulation"""
        return self._manager.move_simulants_to_end

    def register_step_size_modifier(
        self,
        modifier: Callable[[pd.Index[int]], pd.Series[ClockStepSize]],
        requires_columns: list[str] = [],
        requires_values: list[str] = [],
        requires_streams: list[str] = [],
    ) -> None:
        """Registers a step size modifier.

        Parameters
        ----------
        modifier
            Modifier of the step size pipeline. Modifiers can take an index
            and should return a series of step sizes.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the modifier
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the  modifier is called.
        requires_streams
            A list of the randomness streams that need to be properly sourced
            before the modifier is called."""
        return self._manager.register_step_modifier(
            modifier=modifier,
            requires_columns=requires_columns,
            requires_values=requires_values,
            requires_streams=requires_streams,
        )
