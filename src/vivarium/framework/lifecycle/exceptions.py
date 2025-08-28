"""
======================
Lifecycle Exceptions
======================

Exception classes for the lifecycle management system.

"""
from __future__ import annotations

from vivarium.exceptions import VivariumError


class LifeCycleError(VivariumError):
    """Generic error class for the life cycle management system."""

    pass


class InvalidTransitionError(LifeCycleError):
    """Error raised when life cycle ordering contracts are violated."""

    pass


class ConstraintError(LifeCycleError):
    """Error raised when life cycle constraint contracts are violated."""

    pass
