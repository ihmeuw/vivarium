"""
===================
The Population View
===================

The :class:`PopulationView` is a user-facing abstraction that manages read and write access
to the underlying simulation :term:`State Table`. It has two primary responsibilities:

    1. To provide user access to subsets of the simulation state table
       when it is safe to do so.
    2. To allow the user to update the simulation state in a controlled way.

"""
from typing import TYPE_CHECKING, List, Tuple, Union

import pandas as pd

from vivarium.framework.population.exceptions import PopulationError

if TYPE_CHECKING:
    # Cyclic import
    from vivarium.framework.population.manager import PopulationManager


class PopulationView:
    """A read/write manager for the simulation state table.
    It can be used to both read and update the state of the population. A
    PopulationView can only read and write columns for which it is configured.
    Attempts to update non-existent columns are ignored except during
    simulant creation when new columns are allowed to be created.

    Parameters
    ----------
    manager
        The population manager for the simulation.
    columns
        The set of columns this view should have access too.  If empty, this
        view will have access to the entire state table.
    query
        A :mod:`pandas`-style filter that will be applied any time this
        view is read from.

    Notes
    -----
    By default, this view will filter out ``untracked`` simulants unless
    the ``tracked`` column is specified in the initialization arguments.

    """

    def __init__(
        self,
        manager: "PopulationManager",
        view_id: int,
        columns: Union[List[str], Tuple[str]] = (),
        query: str = None,
    ):
        self._manager = manager
        self._id = view_id
        self._columns = list(columns)
        self._query = query

    @property
    def name(self):
        return f"population_view_{self._id}"

    @property
    def columns(self) -> List[str]:
        """The columns that the view can read and update.

        If the view was created with ``None`` as the columns argument, then
        the view will have access to the full table by default. That case
        should be only be used in situations where the full state table is
        actually needed, like for some metrics collection applications.

        """
        if not self._columns:
            return list(self._manager.get_population(True).columns)
        return list(self._columns)

    @property
    def query(self) -> Union[str, None]:
        """A :mod:`pandas` style query to filter the rows of this view.

        This query will be applied any time the view is read. This query may
        reference columns not in the view's columns.

        """
        return self._query

    def subview(self, columns: Union[List[str], Tuple[str]]) -> "PopulationView":
        """Retrieves a new view with a subset of this view's columns.

        Parameters
        ----------
        columns
            The set of columns to provide access to in the subview. Must be
            a proper subset of this view's columns.

        Returns
        -------
        PopulationView
            A new view with access to the requested columns.

        Raises
        ------
        PopulationError
            If the requested columns are not a proper subset of this view's
            columns or no columns are requested.

        Notes
        -----
        Subviews are useful during population initialization. The original
        view may contain both columns that a component needs to create and
        update as well as columns that the component needs to read.  By
        requesting a subview, a component can read the sections it needs
        without running the risk of trying to access uncreated columns
        because the component itself has not created them.

        """

        if not columns or set(columns) - set(self.columns):
            raise PopulationError(
                f"Invalid subview requested.  Requested columns must be a non-empty "
                f"subset of this view's columns.  Requested columns: {columns}, "
                f"Available columns: {self.columns}"
            )
        # Skip constraints for requesting subviews.
        return self._manager._get_view(columns, self.query)

    def get(self, index: pd.Index, query: str = "") -> pd.DataFrame:
        """Select the rows represented by the given index from this view.

        For the rows in ``index`` get the columns from the simulation's
        state table to which this view has access. The resulting rows may be
        further filtered by the view's query and only return a subset
        of the population represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        query
            Additional conditions used to filter the index. These conditions
            will be unioned with the default query of this view.  The query
            provided may use columns that this view does not have access to.

        Returns
        -------
        pandas.DataFrame
            A table with the subset of the population requested.

        Raises
        ------
        PopulationError
            If this view has access to columns that have not yet been created
            and this method is called.  If you see this error, you should
            request a subview with the columns you need read access to.

        See Also
        --------
        :meth:`subview <PopulationView.subview>`

        """
        pop = self._manager.get_population(True).loc[index]

        if not index.empty:
            if self._query:
                pop = pop.query(self._query)
            if query:
                pop = pop.query(query)

        non_existent_columns = set(self.columns) - set(pop.columns)
        if non_existent_columns:
            raise PopulationError(
                f"Requested column(s) {non_existent_columns} not in population table. "
                "This is likely due to a failure to require some columns, randomness "
                "streams, or pipelines when registering a simulant initializer, a value "
                "producer, or a value modifier. NOTE: It is possible for a run to "
                "succeed even if resource requirements were not properly specified in "
                "the simulant initializers or pipeline creation/modification calls. This "
                "success depends on component initialization order which may change in "
                "different run settings."
            )
        return pop.loc[:, self.columns]

    def update(self, population_update: Union[pd.DataFrame, pd.Series]) -> None:
        """Updates the state table with the provided data.

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
        state_table = self._manager.get_population(True)
        population_update = self._format_update_and_check_preconditions(
            population_update,
            state_table,
            self.columns,
            self._manager.creating_initial_population,
            self._manager.adding_simulants,
        )
        if self._manager.creating_initial_population:
            new_columns = list(set(population_update).difference(state_table))
            self._manager._population[new_columns] = population_update[new_columns]
        elif not population_update.empty:
            update_columns = list(set(population_update).intersection(state_table))
            for column in update_columns:
                column_update = self._update_column_and_ensure_dtype(
                    population_update[column],
                    state_table[column],
                    self._manager.adding_simulants,
                )
                self._manager._population[column] = column_update

    def __repr__(self):
        return (
            f"PopulationView(_id={self._id}, _columns={self.columns}, _query={self._query})"
        )

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _format_update_and_check_preconditions(
        population_update: Union[pd.Series, pd.DataFrame],
        state_table: pd.DataFrame,
        view_columns: List[str],
        creating_initial_population: bool,
        adding_simulants: bool,
    ) -> pd.DataFrame:
        """Standardizes the population update format and checks preconditions.

        Managing how values get written to the underlying population state table is critical
        to rule out several categories of error in client simulation code. The state table
        is modified at three different times. In the first, the initial population table
        is being created and new columns are being added to the state table with their
        initial values. In the second, the population manager has added new rows with
        appropriate null values to the state table in response to population creation
        dictated by client code, and population updates are being provided to fill in
        initial values for those new rows. In the final case, state table values for
        existing simulants are being overridden as part of a time step.

        All of these modification scenarios require that certain preconditions are met.
        For all scenarios, we require

            1. The update is a DataFrame or a Series.
            2. If it is a series, it is nameless and this view manages a single column
               or it is named and it's name matches a column in this PopulationView.
            3. The update matches at least one column in this PopulationView.
            4. The update columns are a subset of the columns managed by this
               PopulationView.
            5. The update index is a subset of the existing state table index.
               PopulationViews don't make rows, they just fill them in.

        For initial population creation additional preconditions are documented in
        :meth:`PopulationView._ensure_coherent_initialization`. Outside population
        initialization, we require that all columns in the update to be present in
        the existing state table. When new simulants are added in the middle of the
        simulation, we require that only one component provide updates to a column.

        Parameters
        ----------
        population_update
            The update to the simulation state table.
        state_table
            The existing simulation state table.
        view_columns
            The columns managed by this PopulationView.
        creating_initial_population
            Whether the initial population is being created.
        adding_simulants
            Whether new simulants are currently being initialized.

        Returns
        -------
        pandas.DataFrame
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
            view_columns,
        )

        unknown_simulants = len(population_update.index.difference(state_table.index))
        if unknown_simulants:
            raise PopulationError(
                "Population updates must have an index that is a subset of the current "
                f"population state table. {unknown_simulants} simulants were provided "
                f"in an update with no matching index in the existing table."
            )

        if creating_initial_population:
            PopulationView._ensure_coherent_initialization(population_update, state_table)
        else:
            new_columns = list(set(population_update).difference(state_table))
            if new_columns:
                raise PopulationError(
                    f"Attempting to add new columns {new_columns} to the state table "
                    f"outside the initial population creation phase."
                )

            if adding_simulants:
                state_table_new_simulants = state_table.loc[population_update.index, :]
                conflicting_columns = [
                    column
                    for column in population_update
                    if state_table_new_simulants[column].notnull().any()
                    and not population_update[column].equals(
                        state_table_new_simulants[column]
                    )
                ]
                if conflicting_columns:
                    raise PopulationError(
                        "Two components are providing conflicting initialization data "
                        f"for the state table columns: {conflicting_columns}."
                    )

        return population_update

    @staticmethod
    def _coerce_to_dataframe(
        population_update: Union[pd.Series, pd.DataFrame],
        view_columns: List[str],
    ) -> pd.DataFrame:
        """Coerce all population updates to a :class:`pandas.DataFrame` format.

        Parameters
        ----------
        population_update
            The update to the simulation state table.

        Returns
        -------
        pandas.DataFrame
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
                if len(view_columns) == 1:
                    population_update.name = view_columns[0]
                else:
                    raise PopulationError(
                        "Cannot update with an unnamed pandas series unless there "
                        "is only a single column in the view."
                    )

            population_update = pd.DataFrame(population_update)

        if not set(population_update.columns).issubset(view_columns):
            raise PopulationError(
                f"Cannot update with a DataFrame or Series that contains columns "
                f"the view does not. Dataframe contains the following extra columns: "
                f"{set(population_update.columns).difference(view_columns)}."
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
        population_update: pd.DataFrame, state_table: pd.DataFrame
    ) -> None:
        """Ensure that overlapping population updates have the same information.

        During population initialization, each state table column should be updated by
        exactly one component and each component with an initializer should create at
        least one column. Sometimes components are a little sloppy and provide
        duplicate column information, which we should continue to allow. We want to ensure
        that a column is only getting one set of unique values though.

        Parameters
        ----------
        population_update
            The update to the simulation state table.
        state_table
            The existing simulation state table. When this method is called, the table
            should be in a partially complete state. That is the provided population
            update should carry some new attributes we need to assign.

        Raises
        -----
        PopulationError
            If the population update contains no new information or if it contains
            information in conflict with the existing state table.

        """
        missing_pops = len(state_table.index.difference(population_update.index))
        if missing_pops:
            raise PopulationError(
                f"Components should initialize the same population at the simulation start. "
                f"A component is missing updates for {missing_pops} simulants."
            )
        new_columns = set(population_update).difference(state_table)
        overlapping_columns = set(population_update).intersection(state_table)
        if not new_columns:
            raise PopulationError(
                f"A component is providing a population update for {list(population_update)} "
                "but all provided columns are initialized by other components."
            )
        for column in overlapping_columns:
            if not population_update[column].equals(state_table[column]):
                raise PopulationError(
                    "Two components are providing conflicting initialization data for the "
                    f"{column} state table column."
                )

    @staticmethod
    def _update_column_and_ensure_dtype(
        update: pd.Series,
        existing: pd.Series,
        adding_simulants: bool,
    ) -> pd.Series:
        """Build the updated state table column with an appropriate dtype.

        Parameters
        ----------
        update
            The new column values for a subset of the existing index.
        existing
            The existing column values for all simulants in the state table.
        adding_simulants
            Whether new simulants are currently being initialized.

        Returns
        -------
        pd.Series
            The column with the provided update applied

        """
        # FIXME: This code does not work as described. I'm leaving it here because writing
        #  real dtype checking code is a pain and we never seem to hit the actual edge cases.
        #  I've also seen this error, though I don't have a reproducible and useful example.
        #  I'm reasonably sure what's really being accounted for here is non-nullable columns
        #  that temporarily have null values introduced in the space between rows being
        #  added to the state table and initializers filling them with their first values.
        #  That means the space of dtype casting issues is actually quite small. What should
        #  actually happen in the long term is to separate the population creation entirely
        #  from the mutation of existing state. I.e. there's not an actual reason we need
        #  to do all these sequential operations on a single underlying dataframe during
        #  the creation of new simulants besides the fact that it's the existing
        #  implementation.
        update_values = update.values.copy()
        new_state_table_values = existing.values.copy()
        update_index_positional = existing.index.get_indexer(update.index)

        # Assumes the update index labels can be interpreted as an array position.
        new_state_table_values[update_index_positional] = update_values

        unmatched_dtypes = new_state_table_values.dtype != update_values.dtype
        if unmatched_dtypes and not adding_simulants:
            # This happens when the population is being grown because extending
            # the index forces columns that don't have a natural null type
            # to become 'object'
            raise PopulationError(
                "A component is corrupting the population table by modifying the dtype of "
                f"the {update.name} column from {existing.dtype} to {update.dtype}."
            )
        new_state_table_values = new_state_table_values.astype(update_values.dtype)
        return pd.Series(new_state_table_values, index=existing.index, name=existing.name)
