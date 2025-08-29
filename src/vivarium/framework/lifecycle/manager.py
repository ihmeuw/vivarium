"""
==================
Life Cycle Manager
==================

Manager of ordering- and constraint-based contracts in the simulation

"""
from __future__ import annotations

import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vivarium.framework.lifecycle.constraint_maker import ConstraintMaker
from vivarium.framework.lifecycle.entities import LifeCycle
from vivarium.framework.lifecycle.exceptions import InvalidTransitionError, LifeCycleError
from vivarium.framework.lifecycle.lifecycle_states import INITIALIZATION
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium.framework.event import Event


class LifeCycleManager(Manager):
    """Manages ordering- and constraint-based contracts in the simulation."""

    def __init__(self) -> None:
        self.lifecycle = LifeCycle()
        self._current_state = self.lifecycle.get_state(INITIALIZATION)
        self._current_state_start_time = time.time()
        self._timings: defaultdict[str, list[float]] = defaultdict(list)
        self._make_constraint = ConstraintMaker(self)

    @property
    def name(self) -> str:
        """The name of this component."""
        return "life_cycle_manager"

    @property
    def current_state(self) -> str:
        """The name of the current life cycle state."""
        return self._current_state.name

    @property
    def timings(self) -> dict[str, list[float]]:
        return self._timings

    def add_phase(self, phase_name: str, states: list[str], loop: bool = False) -> None:
        """Add a new phase to the lifecycle.

        Phases must be added in order.

        Parameters
        ----------
        phase_name
            The name of the phase to add.  Phase names must be unique.
        states
            The list of names (in order) of the states that make up the
            life cycle phase.  State names must be unique across the entire
            life cycle.
        loop
            Whether the life cycle phase states loop.

        Raises
        ------
        LifeCycleError
            If the phase or state names are non-unique.
        """
        self.lifecycle.add_phase(phase_name, states, loop)

    def set_state(self, state: str) -> None:
        """Sets the current life cycle state to the provided state.

        Parameters
        ----------
        state
            The name of the state to set.

        Raises
        ------
        LifeCycleError
            If the requested state doesn't exist in the life cycle.
        InvalidTransitionError
            If setting the provided state represents an invalid life cycle
            transition.
        """
        new_state = self.lifecycle.get_state(state)
        if self._current_state.valid_next_state(new_state):
            self._timings[self._current_state.name].append(
                time.time() - self._current_state_start_time
            )
            new_state.enter()
            self._current_state = new_state
            self._current_state_start_time = time.time()
        else:
            raise InvalidTransitionError(
                f"Invalid transition from {self.current_state} "
                f"to {new_state.name} requested."
            )

    def get_state_names(self, phase: str) -> list[str]:
        """Gets all states in the phase in their order of execution.

        Parameters
        ----------
        phase
            The name of the phase to retrieve the states for.

        Returns
        -------
            A list of state names in order of execution.
        """
        return self.lifecycle.get_state_names(phase)

    def add_handlers(self, state_name: str, handlers: list[Callable[[Event], None]]) -> None:
        """Registers a set of functions to be called during a life cycle state.

        This method does not apply any constraints, rather it is used
        to build up an execution order for introspection.

        Parameters
        ----------
        state_name
            The name of the state to register the handlers for.
        handlers
            A list of functions that will execute during the state.
        """
        s = self.lifecycle.get_state(state_name)
        s.add_handlers(handlers)

    def add_constraint(
        self,
        method: Callable[..., Any],
        allow_during: tuple[str, ...] | list[str] = (),
        restrict_during: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Constrains a function to be executable only during certain states.

        Parameters
        ----------
        method
            The method to add constraints to.
        allow_during
            An optional list of life cycle states in which the provided
            method is allowed to be called.
        restrict_during
            An optional list of life cycle states in which the provided
            method is restricted from being called.

        Raises
        ------
        ValueError
            If neither ``allow_during`` nor ``restrict_during`` are provided,
            or if both are provided.
        LifeCycleError
            If states provided as arguments are not in the life cycle.
        ConstraintError
            If a lifecycle constraint has already been applied to the provided
            method.
        """
        if allow_during and restrict_during or not (allow_during or restrict_during):
            raise ValueError(
                'Must provide exactly one of "allow_during" or "restrict_during".'
            )
        unknown_states = (
            set(allow_during).union(restrict_during).difference(self.lifecycle._state_names)
        )
        if unknown_states:
            raise LifeCycleError(
                f"Attempting to constrain {method} with "
                f"states not in the life cycle: {list(unknown_states)}."
            )
        if restrict_during:
            allow_during = [
                s for s in self.lifecycle._state_names if s not in restrict_during
            ]
        if isinstance(allow_during, tuple):
            allow_during = list(allow_during)

        self._make_constraint(method, allow_during)

    def __repr__(self) -> str:
        return f"LifeCycleManager(state={self.current_state})"

    def __str__(self) -> str:
        return str(self.lifecycle)
