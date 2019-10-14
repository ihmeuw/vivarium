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
from math import ceil
from typing import List, Callable, Dict, Any

import pandas as pd

from vivarium.framework.engine import SimulationContext
from vivarium.framework.time import Timedelta, Time
from vivarium.framework.values import Pipeline

from .utilities import run_from_ipython, log_progress


class InteractiveContext(SimulationContext):
    """A simulation context with helper methods for running simulations
    interactively.

    This class should not be instantiated directly. It should be created with a
    call to one of the helper methods provided in this module.
    """

    def __init__(self, *args, setup=True, **kwargs):
        super().__init__(*args, **kwargs)

        if setup:
            self.setup()

    def setup(self):
        super().setup()
        self.initialize_simulants()

    def step(self, step_size: Timedelta = None):
        """Advance the simulation one step.

        Parameters
        ----------
        step_size
            An optional size of step to take. Must be the same type as the
            simulation clock's step size (usually a pandas.Timedelta).
        """
        old_step_size = self._clock.step_size
        if step_size is not None:
            if not isinstance(step_size, type(self._clock.step_size)):
                raise ValueError(f"Provided time must be an instance of {type(self._clock.step_size)}")
            self._clock._step_size = step_size
        super().step()
        self._clock._step_size = old_step_size

    def run(self, with_logging: bool = True) -> int:
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
        return self.run_until(self._clock.stop_time, with_logging=with_logging)

    def run_for(self, duration: Timedelta, with_logging: bool = True) -> int:
        """Run the simulation for the given time duration.

        Parameters
        ----------
        duration
            The length of time to run the simulation for. Should be the same
            type as the simulation clock's step size (usually a pandas
            Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        return self.run_until(self._clock.time + duration, with_logging=with_logging)

    def run_until(self, end_time: Time, with_logging: bool = True) -> int:
        """Run the simulation until the provided end time.

        Parameters
        ----------
        end_time
            The time to run the simulation until. The simulation will run until
            its clock is greater than or equal to the provided end time.
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        if not isinstance(end_time, type(self._clock.time)):
            raise ValueError(f"Provided time must be an instance of {type(self._clock.time)}")

        iterations = int(ceil((end_time - self._clock.time)/self._clock.step_size))
        self.take_steps(number_of_steps=iterations, with_logging=with_logging)
        assert self._clock.time - self._clock.step_size < end_time <= self._clock.time
        return iterations

    def take_steps(self, number_of_steps: int = 1, step_size: Timedelta = None, with_logging: bool = True):
        """Run the simulation for the given number of steps.

        Parameters
        ----------
        number_of_steps
            The number of steps to take.
        step_size
            An optional size of step to take. Must be the same type as the
            simulation clock's step size (usually a pandas.Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.
        """
        if not isinstance(number_of_steps, int):
            raise ValueError('Number of steps must be an integer.')

        if run_from_ipython() and with_logging:
            for _ in log_progress(range(number_of_steps), name='Step'):
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
        """
        return self._population.get_population(untracked)

    def list_values(self) -> List[str]:
        """List the names of all pipelines in the simulation."""
        return list(self._values.keys())

    def get_value(self, value_pipeline_name: str) -> Pipeline:
        """Get the value pipeline associated with the given name."""
        return self._values.get_value(value_pipeline_name)

    def list_events(self) -> List[str]:
        """List all event types registered with the simulation."""
        return self._events.list_events()

    def get_listeners(self, event_type: str) -> List[Callable]:
        """Get all listeners of a particular type of event.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        """
        if event_type not in self._events:
            raise ValueError(f'No event {event_type} in system.')
        return self._events.get_listeners(event_type)

    def get_emitter(self, event_type: str) -> Callable:
        """Get the callable that emits the given type of events.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        """
        if event_type not in self._events:
            raise ValueError(f'No event {event_type} in system.')
        return self._events.get_emitter(event_type)

    def list_components(self) -> Dict[str, Any]:
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

    def __repr__(self):
        return 'InteractiveContext()'
