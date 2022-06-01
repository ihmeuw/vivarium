"""
================================
Population Management Exceptions
================================

Errors related to the mishandling of the simulation state table.
"""

from vivarium.exceptions import VivariumError


class PopulationError(VivariumError):
    """Error raised when the population is invalidly queried or updated."""

    pass
