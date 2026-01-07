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
from vivarium.manager import Interface

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.population import SimulantData
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
        self,
        component: Component | None = None,
        query: str = "",
    ) -> PopulationView:
        """Get a time-varying view of the population state table.

        The requested population view can be used to view the current state or
        to update the state with new values.

        Parameters
        ----------
        component
            The component requesting this view. If None, the view will provide
            read-only access.
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
        return self._manager.get_view(component, query)

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

    def register_initializer(
        self,
        columns: str | Sequence[str] | None,
        initializer: Callable[[SimulantData], None],
        dependencies: Sequence[str | Resource] = (),
    ) -> None:
        """Registers a component's initializer(s) and any columns created by them.

        This does three primary things:
        1. Registers each column's corresponding attribute producer.
        2. Records metadata about which component created which private columns.
        3. Registers the initializer as a resource.

        A `columns` value of None indicates that no private columns are being registered.
        This is useful when a component or manager needs to register an initializer
        that does not create any private columns.

        Parameters
        ----------
        columns
            The state table columns that the given initializer provides the initial
            state information for.
        initializer
            A function that will be called to initialize the state of new simulants.
        dependencies
            The resources that the initializer requires to run. Strings are interpreted
            as attributes.
        """
        self._manager.register_initializer(columns, initializer, dependencies)

    def register_tracked_query(self, query: str) -> None:
        """Updates the default query for all population views.

        Parameters
        ----------
        query
            The new default query to apply to all population views.
        """
        self._manager.register_tracked_query(query)

    def get_tracked_query(self) -> Callable[[], str]:
        return self._manager.get_tracked_query
