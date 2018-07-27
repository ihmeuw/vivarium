"""Vivarium time manager."""
from typing import Union, Callable
from numbers import Number
from datetime import datetime, timedelta

import pandas as pd


_Time = Union[datetime, Number]
_Timedelta = Union[timedelta, Number]


class SimulationClock:
    """Defines a base implementation for a simulation clock."""

    def __init__(self):
        self._time = None
        self._stop_time = None
        self._step_size = None

    @property
    def time(self) -> _Time:
        """The current simulation time."""
        assert self._time is not None, 'No start time provided'
        return self._time

    @property
    def stop_time(self) -> _Time:
        """The time at which the simulation will stop."""
        assert self._stop_time is not None, 'No stop time provided'
        return self._stop_time

    @property
    def step_size(self) -> _Timedelta:
        """The size of the next time step."""
        assert self._step_size is not None, 'No step size provided'
        return self._step_size

    def step_forward(self) -> None:
        """Advances the clock by the current step size."""
        self._time += self.step_size

    def step_backward(self):
        """Rewinds the clock by the current step size."""
        self._time -= self.step_size


class SimpleClock(SimulationClock):
    """A unitless step-count based simulation clock."""

    configuration_defaults = {
        'time': {
            'start': 0,
            'end': 100,
            'step_size': 1,
        }
    }

    def setup(self, builder):
        self._time = builder.configuration.time.start
        self._stop_time = builder.configuration.time.end
        self._step_size = builder.configuration.time.step_size


def get_time_stamp(time):
    return pd.Timestamp(time['year'], time['month'], time['day'])


class DateTimeClock(SimulationClock):
    """A date-time based simulation clock."""

    configuration_defaults = {
        'time': {
            'start': {
                'year': 2005,
                'month': 7,
                'day': 2
            },
            'end': {
                'year': 2010,
                'month': 7,
                'day': 2,
            },
            'step_size': 1,  # Days
        }
    }

    def setup(self, builder):
        time = builder.configuration.time
        self._time = get_time_stamp(time.start)
        self._stop_time = get_time_stamp(time.end)
        self._step_size = pd.Timedelta(days=time.step_size // 1, hours=(time.step_size % 1) * 24)


class TimeInterface:
    def __init__(self, clock: SimulationClock):
        self._clock = clock

    def clock(self) -> Callable[[], Union[datetime, Number]]:
        """Gets a callable that returns the current simulation time."""
        return lambda: self._clock.time

    def step_size(self) -> Callable[[], Union[timedelta, Number]]:
        """Gets a callable that returns the current simulation step size."""
        return lambda: self._clock.step_size
