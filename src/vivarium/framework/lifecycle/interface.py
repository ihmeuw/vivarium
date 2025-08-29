"""
====================
Life Cycle Interface
====================

Interface to the life cycle management system.

"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vivarium.manager import Interface

if TYPE_CHECKING:
    from vivarium.framework.event import Event
    from vivarium.framework.lifecycle.manager import LifeCycleManager


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
