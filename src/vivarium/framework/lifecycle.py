"""
=====================
Life Cycle Management
=====================

The life cycle is a representation of the flow of execution states in a
:mod:`vivarium` simulation. The tools in this model allow a simulation to
formally represent its execution state and use the formal representation to
enforce run-time contracts.

There are two flavors of contracts that this system enforces:

 - **Constraints**: These are contracts around when certain methods,
   particularly those available off the :ref:`Builder <builder_concept>`,
   can be used. For example, :term:`simulants <Simulant>` should only be
   added to the simulation during initial population creation and during
   the main simulation loop, otherwise services necessary for initializing
   that population's attributes may not exist. By applying a constraint,
   we can provide very clear errors about what went wrong, rather than
   a deep and unintelligible stack trace.
 - **Ordering Contracts**: The
   :class:`~vivarium.framework.engine.SimulationContext` will construct
   the formal representation of the life cycle during its initialization.
   Once generated, the context declares as it transitions between
   different lifecycle states and the tools here ensure that only valid
   transitions occur.  These kinds of contracts are particularly useful
   during interactive usage, as they prevent users from, for example,
   running a simulation whose population has not been created.

The tools here also allow for introspection of the simulation life cycle.

"""

from __future__ import annotations

import functools
import textwrap
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vivarium.exceptions import VivariumError
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.event import Event


class LifeCycleError(VivariumError):
    """Generic error class for the life cycle management system."""

    pass


class InvalidTransitionError(LifeCycleError):
    """Error raised when life cycle ordering contracts are violated."""

    pass


class ConstraintError(LifeCycleError):
    """Error raised when life cycle constraint contracts are violated."""

    pass


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
        For setting constraints, see :meth:`LifeCycleInterface.add_constraint`.

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
        self.add_phase("initialization", ["initialization"], loop=False)

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


class ConstraintMaker:
    """Factory for making state-based constraints on component methods."""

    def __init__(self, lifecycle_manager: LifeCycleManager):
        self.lifecycle_manager = lifecycle_manager
        self.constraints: set[str] = set()

    def check_valid_state(
        self, method: Callable[..., Any], permitted_states: list[str]
    ) -> None:
        """Ensures a component method is being called during an allowed state.

        Parameters
        ----------
        method
            The method the constraint is applied to.
        permitted_states
            The states in which the method is permitted to be called.

        Raises
        ------
        ConstraintError
            If the method is being called outside the permitted states.
        """
        current_state = self.lifecycle_manager.current_state
        if current_state not in permitted_states:
            raise ConstraintError(
                f"Trying to call {method} during {current_state},"
                f" but it may only be called during {permitted_states}."
            )

    def constrain_normal_method(
        self, method: Callable[..., Any], permitted_states: list[str]
    ) -> Callable[..., Any]:
        """Only permit a method to be called during the provided states.

        Constraints are applied by dynamically wrapping and binding a method
        to an existing component at run time.

        Parameters
        ----------
        method
            The method to constrain.
        permitted_states
            The life cycle states in which the method can be called.

        Returns
        -------
            The constrained method.
        """

        @functools.wraps(method)
        def _wrapped(*args: Any, **kwargs: Any) -> Any:
            self.check_valid_state(method, permitted_states)
            # Call the __func__ because we're rebinding _wrapped to the method
            # name on the object.  If we called method directly, we'd get
            # two copies of self.
            return method.__func__(*args, **kwargs)  # type: ignore [attr-defined]

        # Invoke the descriptor protocol to bind the wrapped method to the
        # component instance.
        rebound_method: Callable[..., Any] = _wrapped.__get__(method.__self__, method.__self__.__class__)  # type: ignore [attr-defined]
        # Then update the instance dictionary to reflect that the wrapped
        # method is bound to the original name.
        setattr(method.__self__, method.__name__, rebound_method)  # type: ignore [attr-defined]
        return rebound_method

    @staticmethod
    def to_guid(method: Callable[..., Any]) -> str:
        """Convert a method on to a global id.

        Because we dynamically rebind methods, the old ones will get garbage
        collected, making :func:`id` unreliable for checking if a method
        has been constrained before.

        Parameters
        ----------
        method
            The method to convert to a global id.

        Returns
        -------
            The global id of the method.
        """
        return f"{method.__self__.name}.{method.__name__}"  # type: ignore [attr-defined]

    def __call__(
        self, method: Callable[..., Any], permitted_states: list[str]
    ) -> Callable[..., Any]:
        """Only permit a method to be called during the provided states.

        Constraints are applied by dynamically wrapping and binding a method
        to an existing component at run time.

        Parameters
        ----------
        method
            The method to constrain.
        permitted_states
            The life cycle states in which the method can be called.

        Returns
        -------
            The constrained method.

        Raises
        ------
        TypeError
            If an unbound function is supplied for constraint.
        ValueError
            If the provided method is a python "special" method (i.e. a
            method surrounded by double underscores).
        """
        if not hasattr(method, "__self__"):
            raise TypeError(
                "Can only apply constraints to bound object methods. "
                f"You supplied the function {method}."
            )
        name = method.__name__
        if name.startswith("__") and name.endswith("__"):
            raise ValueError(
                "Can only apply constraints to normal object methods. "
                f" You supplied {method}."
            )

        if self.to_guid(method) in self.constraints:
            raise ConstraintError(f"Method {method} has already been constrained.")

        self.constraints.add(self.to_guid(method))
        return self.constrain_normal_method(method, permitted_states)


class LifeCycleManager(Manager):
    """Manages ordering- and constraint-based contracts in the simulation."""

    def __init__(self) -> None:
        self.lifecycle = LifeCycle()
        self._current_state = self.lifecycle.get_state("initialization")
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


class LifeCycleInterface(Interface):
    """Interface to the life cycle management system.

    The life cycle management system allows components to constrain
    methods so that they're only available during certain simulation
    life cycle states.

    """

    def __init__(self, manager: LifeCycleManager):
        self._manager = manager

    def add_handlers(self, state: str, handlers: list[Callable[[Event], None]]) -> None:
        """Registers a set of functions to be called during a life cycle state.

        This method does not apply any constraints, rather it is used
        to build up an execution order for introspection.

        Parameters
        ----------
        state
            The name of the state to register the handlers for.
        handlers
            A list of functions that will execute during the state.
        """
        self._manager.add_handlers(state, handlers)

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
            If a life cycle constraint has already been applied to the
            provided method.
        """
        self._manager.add_constraint(method, allow_during, restrict_during)

    def current_state(self) -> Callable[[], str]:
        """Returns a callable that gets the current simulation lifecycle state.

        Returns
        -------
            A callable that returns the current simulation lifecycle state.
        """
        return lambda: self._manager.current_state
