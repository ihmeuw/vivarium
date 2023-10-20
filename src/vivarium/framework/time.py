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
from typing import TYPE_CHECKING, Callable, Union, List

import pandas as pd

if TYPE_CHECKING:
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
        self.population_view = None
        
    @property
    def name(self):
        return "simulation_clock"
    
    @property
    def columns_created(self) -> List[str]:
        return ["next_event_time", "step_size"]

    @property
    def time(self) -> Time:
        """The current simulation time."""
        assert self._clock_time is not None, "No start time provided"
        return self._clock_time

    @property
    def stop_time(self) -> Time:
        """The time at which the simulation will stop."""
        assert self._stop_time is not None, "No stop time provided"
        return self._stop_time

    @property
    def step_size(self) -> Timedelta:
        """The size of the next time step."""
        assert self._clock_step_size is not None, "No step size provided"
        return self._clock_step_size
    

    def watch_times(self, index: pd.Index) -> pd.Series:
        """The next time each simulant will be updated."""
        assert self.population_view is not None, "No watch times provided"
        return self.population_view.subview(["next_event_time"]).get(index)
    

    def watch_step_sizes(self, index: pd.Index) -> pd.Series:
        """The step size for each simulant."""
        assert self.population_view is not None, "No watch step sizes provided"
        return self.population_view.subview(["step_size"]).get(index)
    
    def setup(self, builder: "Builder"):
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=self.columns_created)
        self.population_view = builder.population.get_view(columns=self.columns_created)
    
    
    def on_initialize_simulants(self, pop_data):
        """Sets the next_event_time and step_size columns for each simulant"""
        watches = pd.DataFrame(
            {
                "next_event_time": [self.time] * len(pop_data.index),
                "step_size": [self.step_size] * len(pop_data.index),
            },
            index=pop_data.index,
        )
        self.population_view.update(watches)
    
    
    def step_backward(self) -> None:
        """Rewinds the clock by the current step size."""
        self._clock_time -= self.step_size
    
    
    def step_forward(self, index: pd.Index) -> None:
        """Advances the clock by the current step size."""
        self._clock_time += self.step_size
        pop_to_update = self.timely_pop(index)
        pop_to_update["next_event_time"] = self.time + pop_to_update["step_size"]
        self.population_view.update(pop_to_update)
        
    
    def timely_pop(self, index: pd.Index = None):
        """Gets population that is aligned with global clock"""
        watches = self.population_view.get(index)
        return watches[watches.next_event_time <= self.time]
    


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
        self._clock_time = builder.configuration.time.start
        self._stop_time = builder.configuration.time.end
        self._clock_step_size = builder.configuration.time.step_size

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
        self._clock_time = get_time_stamp(time.start)
        self._stop_time = get_time_stamp(time.end)
        self._clock_step_size = pd.Timedelta(
            days=time.step_size // 1, hours=(time.step_size % 1) * 24
        )


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
    
    def watch_times(self) -> Callable[[pd.Index], pd.Series]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.watch_times
    
    def watch_step_sizes(self) -> Callable[[pd.Index], pd.Series]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._manager.watch_step_sizes
