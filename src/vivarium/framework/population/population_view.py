"""
===================
The Population View
===================

The :class:`PopulationView` is a user-facing abstraction that manages read and write access
to the underlying simulation :term:`state table`. It has two primary responsibilities:

    1. To provide user access to subsets of the simulation state table
       when it is safe to do so.
    2. To allow the user to update private data in a controlled way.

"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from vivarium.framework.population.exceptions import PopulationError

if TYPE_CHECKING:
    from vivarium.framework.population.manager import PopulationManager
    from vivarium.types import ClassWithName


class PopulationView:
    """A read/write manager for the simulation private data.

    It can be used to both read and update the state of the population. While a
    PopulationView can read any columns, it can only write those columns that the
    component it is attached to created.

    Attempts to update non-existent columns are ignored except during
    simulant creation when new columns are allowed to be created.

    """

    def __init__(
        self,
        manager: PopulationManager,
        component: ClassWithName,
        view_id: int,
        private_columns: Sequence[str] = (),
        query: str = "",
    ):
        """

        Parameters
        ----------
        manager
            The population manager for the simulation.
        component
            The class that created this view.
        view_id
            The unique identifier for this view.
        private_columns
            The columns this view should have write access to.
        query
            A :mod:`pandas`-style filter that will be applied any time this
            view is read from.
        """
        self._manager = manager
        self._component = component
        self._id = view_id
        self.private_columns = list(private_columns)
        self.query = query

    @property
    def name(self) -> str:
        return f"population_view_{self._id}"

    def get(
        self,
        index: pd.Index[int],
        attributes: str | list[str],
        query: str = "",
    ) -> pd.DataFrame:
        """Get a specific subset of this ``PopulationView``.

        For the rows in ``index``, return the ``attributes`` (i.e. columns) from the
        simulation's private data. The resulting rows may be further filtered by the
        view's query and only return a subset of the population represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        attributes
            The columns to retrieve from the population private data.
        query
            Additional conditions used to filter the index. These conditions
            will be unioned with the default query of this view. The query
            provided may not use columns that this view does not have access to.

        Returns
        -------
            A table with the subset of the population requested.

        """

        if isinstance(attributes, str):
            attributes = [attributes]

        combined_query = " and ".join(filter(None, [self.query, query]))
        return self._manager.get_population(
            attributes=attributes,
            index=index,
            query=combined_query,
        )

    def update(self, population_update: pd.Series[Any] | pd.DataFrame) -> None:
        """Updates the private data with the provided data.

        Parameters
        ----------
        population_update
            The data which should be copied into the simulation's state. If
            the update is a :class:`pandas.DataFrame`, it can contain a subset
            of the view's columns but no extra columns. If ``pop`` is a
            :class:`pandas.Series` it must have a name that matches one of
            this view's columns unless the view only has one column in which
            case the Series will be assumed to refer to that regardless of its
            name.

        Raises
        ------
        PopulationError
            If the provided data name or columns do not match columns that
            this view manages or if the view is being updated with a data
            type inconsistent with the original population data.
        """
        if not isinstance(self._manager.population, pd.DataFrame):
            raise PopulationError(
                "Trying to update a population that has not yet been initialized."
            )
        private_data = self._manager.get_private_data(self._component)
        population_update_df: pd.DataFrame = self._format_update_and_check_preconditions(
            population_update,
            private_data,
            self.private_columns,
            self._manager.creating_initial_population,
            self._manager.adding_simulants,
        )
        if self._manager.creating_initial_population:
            new_columns = list(set(population_update_df).difference(private_data))
            self._manager.population[new_columns] = population_update_df[new_columns]
        elif not population_update_df.empty:
            update_columns = list(set(population_update_df).intersection(private_data))
            for column in update_columns:
                column_update = self._update_column_and_ensure_dtype(
                    population_update_df[column],
                    private_data[column],
                    self._manager.adding_simulants,
                )
                self._manager.population[column] = column_update

    def __repr__(self) -> str:
        return f"PopulationView(_id={self._id}, _component={self._component.name}, private_columns={self.private_columns}, query={self.query})"

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _format_update_and_check_preconditions(
        population_update: pd.Series[Any] | pd.DataFrame,
        private_data: pd.DataFrame,
        private_columns: list[str],
        creating_initial_population: bool,
        adding_simulants: bool,
    ) -> pd.DataFrame:
        """Standardizes the population update format and checks preconditions.

        Managing how values get written to the underlying population private data is critical
        to rule out several categories of error in client simulation code. The private data
        is modified at three different times. In the first, the initial population table
        is being created and new columns are being added to the private data with their
        initial values. In the second, the population manager has added new rows with
        appropriate null values to the private data in response to population creation
        dictated by client code, and population updates are being provided to fill in
        initial values for those new rows. In the final case, private data values for
        existing simulants are being overridden as part of a time step.

        All of these modification scenarios require that certain preconditions are met.
        For all scenarios, we require

            1. The update is a DataFrame or a Series.
            2. If it is a series, it is nameless and this view manages a single column
               or it is named and it's name matches a column in this PopulationView.
            3. The update matches at least one column in this PopulationView.
            4. The update columns are a subset of the columns managed by this
               PopulationView.
            5. The update index is a subset of the existing private data index.
               PopulationViews don't make rows, they just fill them in.

        For initial population creation additional preconditions are documented in
        :meth:`PopulationView._ensure_coherent_initialization`. Outside population
        initialization, we require that all columns in the update to be present in
        the existing private data. When new simulants are added in the middle of the
        simulation, we require that only one component provide updates to a column.

        Parameters
        ----------
        population_update
            The update to the private data owned by the component that created this view.
        private_data
            The existing private data owned by the component that created this view.
        private_columns
            The private columns managed by this PopulationView.
        creating_initial_population
            Whether the initial population is being created.
        adding_simulants
            Whether new simulants are currently being initialized.

        Returns
        -------
            The input data formatted as a DataFrame.

        Raises
        ------
        TypeError
            If the population update is not a :class:`pandas.Series` or a
            :class:`pandas.DataFrame`.
        PopulationError
            If the update violates any preconditions relevant to the context in which
            the update is provided (initial population creation, population creation on
            time steps, or population state changes on time steps).

        """
        assert not creating_initial_population or adding_simulants

        population_update = PopulationView._coerce_to_dataframe(
            population_update,
            private_columns,
        )

        unknown_simulants = len(population_update.index.difference(private_data.index))
        if unknown_simulants:
            raise PopulationError(
                "Population updates must have an index that is a subset of the current "
                f"private data. {unknown_simulants} simulants were provided "
                f"in an update with no matching index in the existing table."
            )

        if creating_initial_population:
            PopulationView._ensure_coherent_initialization(population_update, private_data)
        else:
            new_columns = list(set(population_update).difference(private_data))
            if new_columns:
                raise PopulationError(
                    f"Attempting to add new columns {new_columns} to the private data "
                    f"outside the initial population creation phase."
                )

            if adding_simulants:
                private_data_new_simulants = private_data.loc[population_update.index, :]
                conflicting_columns = [
                    column
                    for column in population_update
                    if private_data_new_simulants[column].notnull().any()
                    and not population_update[column].equals(
                        private_data_new_simulants[column]
                    )
                ]
                if conflicting_columns:
                    raise PopulationError(
                        "Two components are providing conflicting initialization data "
                        f"for the private data columns: {conflicting_columns}."
                    )

        return population_update

    @staticmethod
    def _coerce_to_dataframe(
        population_update: pd.Series[Any] | pd.DataFrame,
        private_columns: list[str],
    ) -> pd.DataFrame:
        """Coerce all population updates to a :class:`pandas.DataFrame` format.

        Parameters
        ----------
        population_update
            The update to the private data owned by the component that created this view.
        private_data
            The existing private data owned by the component that created this view.

        Returns
        -------
            The input data formatted as a DataFrame.

        Raises
        ------
        TypeError
            If the population update is not a :class:`pandas.Series` or a
            :class:`pandas.DataFrame`.
        PopulationError
            If the input data is a :class:`pandas.Series` and this :class:`PopulationView`
            manages multiple columns or if the population update contains columns not
            managed by this view.
        """
        if not isinstance(population_update, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "The population update must be a pandas Series or DataFrame. "
                f"A {type(population_update)} was provided."
            )

        if isinstance(population_update, pd.Series):
            if population_update.name is None:
                if len(private_columns) == 1:
                    population_update.name = private_columns[0]
                else:
                    raise PopulationError(
                        "Cannot update with an unnamed pandas series unless there "
                        "is only a single column in the view."
                    )

            population_update = pd.DataFrame(population_update)

        if not set(population_update.columns).issubset(private_columns):
            raise PopulationError(
                f"Cannot update with a DataFrame or Series that contains columns "
                f"the view does not. Dataframe contains the following extra columns: "
                f"{set(population_update.columns).difference(private_columns)}."
            )

        update_columns = list(population_update)
        if not update_columns:
            raise PopulationError(
                "The update method of population view is being called "
                "on a DataFrame with no columns."
            )

        return population_update

    @staticmethod
    def _ensure_coherent_initialization(
        population_update: pd.DataFrame, private_data: pd.DataFrame
    ) -> None:
        """Ensure that overlapping population updates have the same information.

        During population initialization, each private data column should be updated by
        exactly one component and each component with an initializer should create at
        least one column. Sometimes components are a little sloppy and provide
        duplicate column information, which we should continue to allow. We want to ensure
        that a column is only getting one set of unique values though.

        Parameters
        ----------
        population_update
            The update to the private data owned by the component that created this view.
        private_data
            The existing private data owned by the component that created this view.
            When this method is called, the table should be in a partially complete
            state. That is the provided population update should carry some new
            attributes we need to assign.

        Raises
        -----
        PopulationError
            If the population update contains no new information or if it contains
            information in conflict with the existing private data.
        """
        missing_pops = len(private_data.index.difference(population_update.index))
        if missing_pops:
            raise PopulationError(
                f"Components should initialize the same population at the simulation start. "
                f"A component is missing updates for {missing_pops} simulants."
            )
        new_columns = set(population_update).difference(private_data)
        overlapping_columns = set(population_update).intersection(private_data)
        if not new_columns:
            raise PopulationError(
                f"A component is providing a population update for {list(population_update)} "
                "but all provided columns are initialized by other components."
            )
        for column in overlapping_columns:
            if not population_update[column].equals(private_data[column]):
                raise PopulationError(
                    "Two components are providing conflicting initialization data for the "
                    f"{column} private data column."
                )

    @staticmethod
    def _update_column_and_ensure_dtype(
        update: pd.Series[Any],
        existing: pd.Series[Any],
        adding_simulants: bool,
    ) -> pd.Series[Any]:
        """Build the updated private data column with an appropriate dtype.

        Parameters
        ----------
        update
            The new column values for a subset of the existing index.
        existing
            The existing column values for all simulants in the private data.
        adding_simulants
            Whether new simulants are currently being initialized.

        Returns
        -------
            The column with the provided update applied
        """
        # FIXME: This code does not work as described. I'm leaving it here because writing
        #  real dtype checking code is a pain and we never seem to hit the actual edge cases.
        #  I've also seen this error, though I don't have a reproducible and useful example.
        #  I'm reasonably sure what's really being accounted for here is non-nullable columns
        #  that temporarily have null values introduced in the space between rows being
        #  added to the private data and initializers filling them with their first values.
        #  That means the space of dtype casting issues is actually quite small. What should
        #  actually happen in the long term is to separate the population creation entirely
        #  from the mutation of existing state. I.e. there's not an actual reason we need
        #  to do all these sequential operations on a single underlying dataframe during
        #  the creation of new simulants besides the fact that it's the existing
        #  implementation.
        update_values = update.array.copy()
        new_private_data_values = existing.array.copy()
        update_index_positional = existing.index.get_indexer(update.index)  # type: ignore [no-untyped-call]

        # Assumes the update index labels can be interpreted as an array position.
        new_private_data_values[update_index_positional] = update_values

        unmatched_dtypes = new_private_data_values.dtype != update_values.dtype
        if unmatched_dtypes and not adding_simulants:
            # This happens when the population is being grown because extending
            # the index forces columns that don't have a natural null type
            # to become 'object'
            raise PopulationError(
                "A component is corrupting the population table by modifying the dtype of "
                f"the {update.name} column from {existing.dtype} to {update.dtype}."
            )
        new_private_data_values = new_private_data_values.astype(update_values.dtype)
        new_private_data: pd.Series[Any] = pd.Series(
            new_private_data_values, index=existing.index, name=existing.name
        )
        return new_private_data
