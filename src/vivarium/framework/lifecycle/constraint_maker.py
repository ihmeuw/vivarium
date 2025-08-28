"""
================
Constraint Maker
================

Factory for making state-based constraints on component methods.

"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vivarium.framework.lifecycle.exceptions import ConstraintError

if TYPE_CHECKING:
    from vivarium.framework.lifecycle.manager import LifeCycleManager


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
