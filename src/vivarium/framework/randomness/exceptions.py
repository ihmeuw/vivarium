"""
============================
Randomness System Exceptions
============================

Errors related to improper use of the Randomness system.

"""
from vivarium.exceptions import VivariumError


class RandomnessError(VivariumError):
    """Raised for inconsistencies in random number and choice generation."""

    pass
