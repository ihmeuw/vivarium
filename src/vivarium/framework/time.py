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
from typing import Callable, Union, List

import pandas as pd
from vivarium.framework.engine import Builder

from vivarium.manager import Manager

Time = Union[pd.Timestamp, datetime, Number]
Timedelta = Union[pd.Timedelta, timedelta, Number]

class SimulationClock(Manager):
    """A base clock that includes global clock and a pandas series of clocks for each simulant"""
    def __init__(self):
        self._clock_time = None
        self._stop_time = None
        self._clock_step_size = None
        self._watch_times = None
        self._watch_step_sizes = None
        
    @property
    def name(self):
        return "multi_clock"
    
    @property
    def columns_created(self) -> List[str]:
        return ["next_event_time", "step_size"]

    @property
    def clock_time(self) -> Time:
        """The current simulation time."""
        assert self._clock_time is not None, "No start time provided"
        return self._clock_time

    @property
    def stop_time(self) -> Time:
        """The time at which the simulation will stop."""
        assert self._clock_stop_time is not None, "No stop time provided"
        return self._clock_stop_time

    @property
    def clock_step_size(self) -> Timedelta:
        """The size of the next time step."""
        assert self._clock_step_size is not None, "No step size provided"
        return self._clock_step_size
    
    @property
    def watch_times(self) -> pd.Series:
        """The next time each simulant will be updated."""
        assert self._watch_times is not None, "No watch times provided"
        return self._watch_times
    
    @property
    def watch_step_sizes(self) -> pd.Series:
        """The step size for each simulant."""
        assert self._watch_step_sizes is not None, "No watch step sizes provided"
        return self._watch_step_sizes
    
    def setup(self, builder: Builder):
        builder.population.register_simulant_initializer(self.on_initialize_simulants, creates_columns=self.columns_created)
        self.population_view = builder.population.get_view(columns=self.columns_created)
    
    def step_forward(self) -> None:
        """Advances the clock by the current step size."""
        self._clock_time += self.clock_step_size
        pop_to_update = self.timely_pop()
        self._watch_times.loc[pop_to_update] += self.watch_step_sizes.loc[pop_to_update]

    def step_backward(self):
        """Rewinds the clock by the current step size."""
        self._clock_time -= self.clock_step_size
        pop_to_update = self.timely_pop()
        self._watch_times.loc[pop_to_update] -= self.watch_step_sizes.loc[pop_to_update]
    
    def timely_pop(self):
        """Gets population that is aligned with global clock"""
        watch_times = self.watch_times
        return watch_times.index[watch_times == self.clock_time]
    
    def on_initialize_simulants(self, pop_data):
        """Sets the next_event_time and step_size columns for each simulant"""
        watches = pd.DataFrame(
            {
                "next_event_time": [self.clock_time] * len(pop_data.index),
                "step_size": [self.clock_step_size] * len(pop_data.index),
            },
            index=pop_data.index,
        )
        self.population_view.update(watches)


class SimpleClock(SimulationClock):
    """A unitless step-count based simulation clock."""

    configuration_defaults = {
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
        super.setup(builder)    
        self._time = builder.configuration.time.start
        self._stop_time = builder.configuration.time.end
        self._step_size = builder.configuration.time.step_size
        self._watch_times = self.population_view.get("next_event_time")
        self._watch_step_sizes = self.population_view.get("step_size")

    def __repr__(self):
        return "SimpleClock()"


def get_time_stamp(time):
    return pd.Timestamp(time["year"], time["month"], time["day"])


class DateTimeClock(SimulationClock):
    """A date-time based simulation clock."""

    configuration_defaults = {
        "time": {
            "start": {"year": 2005, "month": 7, "day": 2},
            "end": {
                "year": 2010,
                "month": 7,
                "day": 2,
            },
            "step_size": 1,  # Days
        }
    }

    @property
    def name(self):
        return "datetime_clock"

    def setup(self, builder):
        super().setup(builder)
        time = builder.configuration.time
        self._time = get_time_stamp(time.start)
        self._stop_time = get_time_stamp(time.end)
        self._step_size = pd.Timedelta(
            days=time.step_size // 1, hours=(time.step_size % 1) * 24
        )
        self._watch_times = self.population_view.get("next_event_time")
        self._watch_step_sizes = self.population_view.get("step_size")

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
    
    def watch_times(self) -> Callable[[], pd.Series]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.watch_times
    
    def watch_step_sizes(self) -> Callable[[], pd.Series]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.watch_step_sizes
