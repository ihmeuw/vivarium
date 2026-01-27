"""
==============
Time Interface
==============

This module provides an interface to the various types of
:class:`simulation clocks <vivarium.framework.time.manager.SimulationClock>` for
use in ``vivarium``.

For more information about time in the simulation, see the associated
:ref:`concept note <time_concept>`.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import pandas as pd

from vivarium.types import ClockStepSize, ClockTime

if TYPE_CHECKING:
    from vivarium.framework.resource import Resource
    from vivarium.framework.time.manager import SimulationClock

from vivarium.manager import Interface


class TimeInterface(Interface):
    """Public interface for the simulation time management system."""

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
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Registers a step size modifier.

        Parameters
        ----------
        modifier
            Modifier of the step size pipeline. Modifiers can take an index
            and should return a series of step sizes.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called. This is a list of strings, pipelines,
            or randomness streams.

        """
        return self._manager.register_step_modifier(
            modifier=modifier, required_resources=required_resources
        )
