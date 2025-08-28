"""
=====================
Life Cycle Management
=====================

This package provides tools for managing the simulation lifecycle.

"""

from vivarium.framework.lifecycle.entities import (
    COLLECT_METRICS,
    INITIALIZATION,
    POPULATION_CREATION,
    POST_SETUP,
    REPORT,
    SETUP,
    SIMULATION_END,
    TIME_STEP,
    TIME_STEP_CLEANUP,
    TIME_STEP_PREPARE,
)
from vivarium.framework.lifecycle.exceptions import ConstraintError
from vivarium.framework.lifecycle.manager import LifeCycleInterface, LifeCycleManager
