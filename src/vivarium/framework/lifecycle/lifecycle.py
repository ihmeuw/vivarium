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
import time
from collections import defaultdict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vivarium.exceptions import VivariumError
from vivarium.framework.lifecycle.entities import (
    LifeCycle,
    LifeCyclePhase,
    LifeCycleState,
)
from vivarium.framework.lifecycle.exceptions import (
    ConstraintError,
    InvalidTransitionError,
    LifeCycleError,
)
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.event import Event


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
