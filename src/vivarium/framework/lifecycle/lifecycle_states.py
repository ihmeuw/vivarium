"""
================
Lifecycle States
================

This module defines constants representing the states in the simulation lifecycle.

"""


# Initialization states
INITIALIZATION = "initialization"
"""The name of the initialization state in the simulation lifecycle."""

# Setup states
SETUP = "setup"
"""The name of the setup state in the simulation lifecycle."""

POST_SETUP = "post_setup"
"""The name of the post-setup state in the simulation lifecycle."""

POPULATION_CREATION = "population_creation"
"""The name of the population creation state in the simulation lifecycle."""

#  Main loop states
TIME_STEP_PREPARE = "time_step__prepare"
"""The name of the time step preparation state in the simulation lifecycle."""
TIME_STEP = "time_step"
"""The name of the time step state in the simulation lifecycle."""
TIME_STEP_CLEANUP = "time_step__cleanup"
"""The name of the time step cleanup state in the simulation lifecycle."""
COLLECT_METRICS = "collect_metrics"
"""The name of the collect metrics state in the simulation lifecycle."""

# Simulation end states
SIMULATION_END = "simulation_end"
"""The name of the simulation end state in the simulation lifecycle."""
REPORT = "report"
"""The name of the report state in the simulation lifecycle."""
