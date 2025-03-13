"""
==========================
Vivarium Interactive Tools
==========================

This module provides an interface for interactive simulation usage. The main
part is the :class:`InteractiveContext`, a sub-class of the main simulation
object in ``vivarium`` that has been extended to include convenience
methods for running and exploring the simulation in an interactive setting.

See the associated tutorials for :ref:`running <interactive_tutorial>` and
:ref:`exploring <exploration_tutorial>` for more information.

"""
from __future__ import annotations

from collections.abc import Callable
from math import ceil
from typing import Any

import pandas as pd

from vivarium.framework.engine import SimulationContext
from vivarium.framework.event import Event
from vivarium.framework.values import Pipeline
from vivarium.interface.utilities import log_progress, run_from_ipython
from vivarium.types import ClockStepSize, ClockTime


class InteractiveContext(SimulationContext):
    """A simulation context with helper methods for running simulations interactively."""

    def __init__(self, *args: Any, setup: bool = True, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        if setup:
            self.setup()

    @property
    def current_time(self) -> ClockTime:
        """Returns the current simulation time."""
        return self._clock.time

    def setup(self) -> None:
        super().setup()
        self.initialize_simulants()

    def step(self, step_size: ClockStepSize | None = None) -> None:
        """Advance the simulation one step.

        Parameters
        ----------
        step_size
            An optional size of step to take. Must be compatible with the
            simulation clock's step size (usually a pandas.Timedelta).
        """
        old_step_size = self._clock._clock_step_size
        if step_size is not None:
            if not (
                isinstance(step_size, type(self._clock.step_size))
                or isinstance(self._clock.step_size, type(step_size))
            ):
                raise ValueError(
                    f"Provided time must be compatible with {type(self._clock.step_size)}"
                )
            self._clock._clock_step_size = step_size
        super().step()
        self._clock._clock_step_size = old_step_size

    def run(self, with_logging: bool = True) -> None:  # type: ignore [override]
        """Run the simulation for the duration specified in the configuration.

        Parameters
        ----------
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        self.run_until(self._clock.stop_time, with_logging=with_logging)

    def run_for(self, duration: ClockStepSize, with_logging: bool = True) -> None:
        """Run the simulation for the given time duration.

        Parameters
        ----------
        duration
            The length of time to run the simulation for. Should be compatible
            with the simulation clock's step size (usually a pandas
            Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        self.run_until(self._clock.time + duration, with_logging=with_logging)  # type: ignore [operator]

    def run_until(self, end_time: ClockTime, with_logging: bool = True) -> None:
        """Run the simulation until the provided end time.

        Parameters
        ----------
        end_time
            The time to run the simulation until. The simulation will run until
            its clock is greater than or equal to the provided end time. Must be
            compatible with the simulation clock's step size (usually a pandas.Timestamp)

        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        if not (
            isinstance(end_time, type(self._clock.time))
            or isinstance(self._clock.time, type(end_time))
        ):
            raise ValueError(
                f"Provided time must be compatible with {type(self._clock.time)}"
            )

        iterations = int(ceil((end_time - self._clock.time) / self._clock.step_size))  # type: ignore [operator, arg-type]
        self.take_steps(number_of_steps=iterations, with_logging=with_logging)
        assert self._clock.time - self._clock.step_size < end_time <= self._clock.time  # type: ignore [operator]
        print("Simulation complete after", iterations, "iterations")

    def take_steps(
        self,
        number_of_steps: int = 1,
        step_size: ClockStepSize | None = None,
        with_logging: bool = True,
    ) -> None:
        """Run the simulation for the given number of steps.

        Parameters
        ----------
        number_of_steps
            The number of steps to take.
        step_size
            An optional size of step to take. Must be compatible with the
            simulation clock's step size (usually a pandas.Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.
        """
        if not isinstance(number_of_steps, int):
            raise ValueError("Number of steps must be an integer.")

        if run_from_ipython() and with_logging:
            for _ in log_progress(range(number_of_steps), name="Step"):
                self.step(step_size)
        else:
            for _ in range(number_of_steps):
                self.step(step_size)

    def get_population(self, untracked: bool = False) -> pd.DataFrame:
        """Get a copy of the population state table.

        Parameters
        ----------
        untracked
            Whether or not to return simulants who are no longer being tracked
            by the simulation.

        Returns
        -------
            The population state table.
        """
        return self._population.get_population(untracked)

    def list_values(self) -> list[str]:
        """List the names of all pipelines in the simulation."""
        return list(self._values.keys())

    def get_value(self, value_pipeline_name: str) -> Pipeline:
        """Get the value pipeline associated with the given name."""
        if value_pipeline_name not in self.list_values():
            raise ValueError(f"No value pipeline '{value_pipeline_name}' registered.")
        return self._values.get_value(value_pipeline_name)

    def list_events(self) -> list[str]:
        """List all event types registered with the simulation."""
        return self._events.list_events()

    def get_listeners(self, event_type: str) -> dict[int, list[Callable[[Event], None]]]:
        """Get all listeners of a particular type of event.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        Returns
        -------
            A dictionary that maps each priority level of the named event's
            listeners to a list of listeners at that level.
        """
        if event_type not in self._events:
            raise ValueError(f"No event {event_type} in system.")
        return self._events.get_listeners(event_type)

    def get_emitter(
        self, event_type: str
    ) -> Callable[[pd.Index[int], dict[str, Any] | None], Event]:
        """Get the callable that emits the given type of events.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        Returns
        -------
            The callable that emits the named event.
        """
        if event_type not in self._events:
            raise ValueError(f"No event {event_type} in system.")
        return self._events.get_emitter(event_type)

    def list_components(self) -> dict[str, Any]:
        """Get a mapping of component names to components currently in the simulation.

        Returns
        -------
            A dictionary mapping component names to components.
        """
        return self._component_manager.list_components()

    def get_component(self, name: str) -> Any:
        """Get the component in the simulation that has ``name``, if present.
        Names are guaranteed to be unique.

        Parameters
        ----------
        name
            A component name.

        Returns
        -------
            A component that has the name ``name`` else None.
        """
        return self._component_manager.get_component(name)

    def print_initializer_order(self) -> None:
        """Print the order in which population initializers are called."""
        initializers = []
        for r in self._resource.get_population_initializers():
            name = r.__name__
            if hasattr(r, "__self__"):
                obj = r.__self__
                initializers.append(f"{obj.__class__.__name__}({obj.name}).{name}")
            else:
                initializers.append(f"Unbound function {name}")
        print("\n".join(initializers))

    def print_lifecycle_order(self) -> None:
        """Print the order of lifecycle events (including user event handlers)."""
        print(self._lifecycle)

    def __repr__(self) -> str:
        return "InteractiveContext()"
