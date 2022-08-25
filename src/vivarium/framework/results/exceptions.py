"""
==========================
Results System Exceptions
==========================

Errors related to the results system
"""

from vivarium.exceptions import VivariumError


class ResultsConfigurationError(VivariumError):
    """Error raised when the results stratifications are improperly configured."""

    pass
