"""
================================
The Population Management System
================================

This subpackage provides tools for managing the :term:`state table <State Table>`
in a :mod:`vivarium` simulation, which is the record of all simulants in a
simulation and their state. Its main tasks are managing the creation of new
simulants and providing the ability for components to view and update simulant
state safely during runtime.

"""
from vivarium.framework.population.exceptions import PopulationError
from vivarium.framework.population.manager import (
    PopulationInterface,
    PopulationManager,
    SimulantData,
)
from vivarium.framework.population.population_view import PopulationView
