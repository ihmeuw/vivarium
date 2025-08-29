"""
==================
Lifecycle Entities
==================

Core entity classes for the lifecycle management system.

"""
from __future__ import annotations

import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING

from vivarium.framework.lifecycle.exceptions import LifeCycleError
from vivarium.framework.lifecycle.lifecycle_states import INITIALIZATION

if TYPE_CHECKING:
    from vivarium.framework.event import Event


class LifeCycleState:
    """A representation of a simulation run state."""

    def __init__(self, name: str):
        self._name = name
        self._next: LifeCycleState | None = None
        self._loop_next: LifeCycleState | None = None
        self._entrance_count = 0
        self._handlers: list[str] = []

    @property
    def name(self) -> str:
        """The name of the lifecycle state."""
        return self._name

    @property
    def entrance_count(self) -> int:
        """The number of times this state has been entered."""
        return self._entrance_count

    def add_next(self, next_state: LifeCycleState, loop: bool = False) -> None:
        """Link this state to the next state in the simulation life cycle.

        States are linked together and used to ensure that the simulation
        life cycle proceeds in the proper order.  A life cycle state can be
        bound to two ``next`` states to allow for loops in the life cycle and
        both are considered valid when checking for valid state transitions.
        The first represents the linear progression through the simulation,
        while the second represents a loop in the life cycle.

        Parameters
        ----------
        next_state
            The next state in the simulation life cycle.
        loop
            Whether the provided state is the linear next state or a loop
            back to a previous state in the life cycle.
        """
        if loop:
            self._loop_next = next_state
        else:
            self._next = next_state

    def valid_next_state(self, state: LifeCycleState | None) -> bool:
        """Check if the provided state is valid for a life cycle transition.

        Parameters
        ----------
        state
            The state to check.

        Returns
        -------
            Whether the state is valid for a transition.
        """
        return (state is None and state is self._next) or (
            state is not None and (state is self._next or state is self._loop_next)
        )

    def enter(self) -> None:
        """Marks an entrance into this state."""
        self._entrance_count += 1

    def add_handlers(self, handlers: list[Callable[[Event], None]]) -> None:
        """Registers a set of functions that will be executed during the state.

        The primary use case here is for introspection and reporting.
        For setting constraints, see
        :meth:`vivarium.framework.lifecycle.interface.LifeCycleInterface.add_constraint`.

        Parameters
        ----------
        handlers
            The set of functions that will be executed during this state.
        """
        for h in handlers:
            name = h.__name__
            if hasattr(h, "__self__"):
                obj = h.__self__
                self._handlers.append(f"{obj.__class__.__name__}({obj.name}).{name}")
            else:
                self._handlers.append(f"Unbound function {name}")

    def __repr__(self) -> str:
        return f"LifeCycleState(name={self.name})"

    def __str__(self) -> str:
        return "\n\t".join([self.name] + self._handlers)


class LifeCyclePhase:
    """A representation of a distinct lifecycle phase in the simulation.

    A lifecycle phase is composed of one or more unique lifecycle states.
    There is exactly one state within the phase which serves as a valid
    exit point from the phase.  The states may operate in a loop.

    """

    def __init__(self, name: str, states: list[str], loop: bool):
        self._name = name
        self._states = [LifeCycleState(states[0])]
        self._loop = loop
        for s in states[1:]:
            self._states.append(LifeCycleState(s))
            self._states[-2].add_next(self._states[-1])
        if self._loop:
            self._states[-1].add_next(self._states[0], loop=True)

    @property
    def name(self) -> str:
        """The name of this life cycle phase."""
        return self._name

    @property
    def states(self) -> tuple[LifeCycleState, ...]:
        """The states in this life cycle phase in order of execution."""
        return tuple(self._states)

    def add_next(self, phase: LifeCyclePhase) -> None:
        """Link the provided phase as the next phase in the life cycle."""
        self._states[-1].add_next(phase._states[0])

    def get_state(self, state_name: str) -> LifeCycleState:
        """Retrieve a life cycle state by name from the phase."""
        return [s for s in self._states if s.name == state_name].pop()

    def __contains__(self, state_name: str) -> bool:
        return bool([s for s in self._states if s.name == state_name])

    def __repr__(self) -> str:
        return f"LifeCyclePhase(name={self.name}, states={[s.name for s in self.states]})"

    def __str__(self) -> str:
        out = self.name
        if self._loop:
            out += "*"
        out += "\n" + textwrap.indent("\n".join([str(state) for state in self.states]), "\t")
        return out


class LifeCycle:
    """A concrete representation of the flow of simulation execution states."""

    def __init__(self) -> None:
        self._state_names: set[str] = set()
        self._phase_names: set[str] = set()
        self._phases: list[LifeCyclePhase] = []
        self.add_phase("initialization", [INITIALIZATION], loop=False)

    def add_phase(self, phase_name: str, states: list[str], loop: bool) -> None:
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
        self._validate(phase_name, states)

        new_phase = LifeCyclePhase(phase_name, states, loop)
        if self._phases:
            self._phases[-1].add_next(new_phase)

        self._state_names.update(states)
        self._phase_names.add(phase_name)
        self._phases.append(new_phase)

    def get_state(self, state_name: str) -> LifeCycleState:
        """Retrieve a life cycle state from the life cycle.

        Parameters
        ----------
        state_name
            The name of the state to retrieve

        Returns
        -------
            The requested state.

        Raises
        ------
        LifeCycleError
            If the requested state does not exist.
        """
        if state_name not in self:
            raise LifeCycleError(f"Attempting to look up non-existent state {state_name}.")
        phase = [p for p in self._phases if state_name in p].pop()
        return phase.get_state(state_name)

    def get_state_names(self, phase_name: str) -> list[str]:
        """Retrieve the names of all states in the provided phase.

        Parameters
        ----------
        phase_name
            The name of the phase to retrieve the state names from.

        Return
        ------
            The state names in the provided phase.

        Raises
        ------
        LifeCycleError
            If the phase does not exist in the life cycle.
        """
        if phase_name not in self._phase_names:
            raise LifeCycleError(
                f"Attempting to look up states from non-existent phase {phase_name}."
            )
        phase = [p for p in self._phases if p.name == phase_name].pop()
        return [s.name for s in phase.states]

    def _validate(self, phase_name: str, states: list[str]) -> None:
        """Validates that a phase and set of states are unique."""
        if phase_name in self._phase_names:
            raise LifeCycleError(
                f"Lifecycle phase names must be unique. You're attempting "
                f"to add {phase_name} but it already exists."
            )
        if len(states) != len(set(states)):
            raise LifeCycleError(
                f"Attempting to create a life cycle phase with duplicate state names. "
                f"States: {states}"
            )
        duplicates = self._state_names.intersection(states)
        if duplicates:
            raise LifeCycleError(
                f"Lifecycle state names must be unique.  You're attempting "
                f"to add {duplicates} but they already exist."
            )

    def __contains__(self, state_name: str) -> bool:
        return state_name in self._state_names

    def __repr__(self) -> str:
        return f"LifeCycle(phases={self._phase_names})"

    def __str__(self) -> str:
        return "\n".join([str(phase) for phase in self._phases])
