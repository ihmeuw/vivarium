"""
====================
Population Interface
====================

This module provides a :class:`PopulationInterface <PopulationInterface>` class with
methods to initialize simulants and get a population view.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from vivarium.framework.population.population_view import PopulationView
from vivarium.framework.resource import Resource
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.population.manager import PopulationManager


class PopulationInterface(Interface):
    """Provides access to the system for reading and updating the population.

    The most important aspect of the simulation state is the ``population
    table`` or ``state table``. It is a table with a row for every
    individual or cohort (referred to as a simulant) being simulated and a
    column for each of the attributes of the simulant being modeled. All
    access to the state table is mediated by
    :class:`population views <vivarium.framework.population.population_view.PopulationView>`,
    which may be requested from this system during setup time.

    """

    def __init__(self, manager: PopulationManager):
        self._manager = manager

    def get_view(
        self, private_columns: str | Sequence[str], query: str = ""
    ) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        Parameters
        ----------
        private_columns
            The private columns created by the component requesting this view.
        query
            A filter on the population state. This filters out particular
            simulants (rows in the state table) based on their current state.
            The query should be provided in a way that is understood by the
            :meth:`pandas.DataFrame.query` method and may reference state
            table columns not requested in the ``private_columns`` argument.

        Returns
        -------
            A filtered view of the requested columns of the population state table.
        """
        return self._manager.get_view(private_columns, query)

    def get_simulant_creator(self) -> Callable[[int, dict[str, Any] | None], pd.Index[int]]:
        """Gets a function that can generate new simulants.

        The creator function takes the number of simulants to be created as it's
        first argument and a dict population configuration that will be available
        to simulant initializers as it's second argument. It generates the new rows
        in the population state table and then calls each initializer
        registered with the population system with a data
        object containing the state table index of the new simulants, the
        configuration info passed to the creator, the current simulation
        time, and the size of the next time step.

        Returns
        -------
           The simulant creator function.
        """
        return self._manager.get_simulant_creator()

    def initializes_simulants(
        self,
        component: Component | Manager,
        creates_columns: str | Sequence[str] = (),
        requires_columns: str | Sequence[str] = (),
        requires_values: str | Sequence[str] = (),
        requires_streams: str | Sequence[str] = (),
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a source of initial state information for new simulants.

        Parameters
        ----------
        component
            The component or manager that will add or update initial state
            information about new simulants.
        creates_columns
            The state table columns that the given initializer
            provides the initial state information for.
        requires_columns
            The state table columns that already need to be present
            and populated in the state table before the provided initializer
            is called.
        requires_values
            The value pipelines that need to be properly sourced
            before the provided initializer is called.
        requires_streams
            The randomness streams necessary to initialize the
            simulant attributes.
        required_resources
            The resources that the initializer requires to run. Strings are
            interpreted as column names, and Pipelines and RandomnessStreams
            are interpreted as value pipelines and randomness streams,
        """
        self._manager.register_simulant_initializer(
            component,
            creates_columns,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
        )

    def register_source_columns(self, component: Component | Manager) -> None:
        """Registers the source columns created by a component or manager.

        Parameters
        ----------
        component
            The component or manager that is registering its source columns.
        """
        self._manager.register_source_columns(component)
