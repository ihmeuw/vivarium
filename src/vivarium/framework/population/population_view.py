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
from typing import List, Tuple, Union, TYPE_CHECKING

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
            columns.

        Notes
        -----
        Subviews are useful during population initialization. The original
        view may contain both columns that a component needs to create and
        update as well as columns that the component needs to read.  By
        requesting a subview, a component can read the sections it needs
        without running the risk of trying to access uncreated columns
        because the component itself has not created them.

        """
        if set(columns) - set(self.columns):
            raise PopulationError(
                f"Invalid subview requested.  Requested columns must be a subset of this "
                f"view's columns.  Requested columns: {columns}, Available columns: {self.columns}"
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

        if not self._columns:
            return pop
        else:
            columns = self._columns
            non_existent_columns = set(columns) - set(pop.columns)
            if non_existent_columns:
                raise PopulationError(
                    f"Requested column(s) {non_existent_columns} not in population table. This is "
                    "likely due to a failure to require some columns, randomness streams, or "
                    "pipelines when registering a simulant initializer, a value producer, or a "
                    "value modifier. NOTE: It is possible for a run to succeed even if resource "
                    "requirements were not properly specified in the simulant initializers or "
                    "pipeline creation/modification calls. This success depends on component "
                    "initialization order which may change in different run settings."
                )
            else:
                return pop.loc[:, columns]

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
        population_update = self._coerce_to_dataframe(population_update)
        affected_columns = set(population_update.columns)

        if population_update.empty and not affected_columns.difference(self._manager.columns):
            return

        state_table = self._manager.get_population(True)
        if not self._manager.growing:
            affected_columns = affected_columns.intersection(state_table.columns)

        for affected_column in affected_columns:
            if affected_column in state_table:
                if population_update.empty:
                    continue

                new_state_table_values = state_table[affected_column].values
                update_values = population_update[affected_column].values
                new_state_table_values[population_update.index] = update_values

                if new_state_table_values.dtype != update_values.dtype:
                    # This happens when the population is being grown because extending
                    # the index forces columns that don't have a natural null type
                    # to become 'object'
                    if not self._manager.growing:
                        raise PopulationError(
                            "Component corrupting population table. "
                            f"Column name: {affected_column} "
                            f"Old column type: {new_state_table_values.dtype} "
                            f"New column type: {update_values.dtype}"
                        )
                    new_state_table_values = new_state_table_values.astype(
                        update_values.dtype
                    )
            else:
                new_state_table_values = population_update[affected_column].values
            self._manager._population[affected_column] = new_state_table_values

    def _coerce_to_dataframe(
        self,
        population_update: Union[pd.Series, pd.DataFrame],
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
        PopulationError
            If the input data is a :class:`pandas.Series` and this :class:`PopulationView`
            manages multiple columns or if the population update contains columns not
            managed by this view.

        """
        if isinstance(population_update, pd.Series):
            if population_update.name is None:
                if len(self.columns) == 1:
                    population_update.name = self.columns[0]
                else:
                    raise PopulationError(
                        "Cannot update with an unnamed pandas series unless there "
                        "is only a single column in the view."
                    )

            population_update = pd.DataFrame(population_update)

        if not set(population_update.columns).issubset(self.columns):
            raise PopulationError(
                f"Cannot update with a DataFrame or Series that contains columns "
                f"the view does not. Dataframe contains the following extra columns: "
                f"{set(population_update.columns).difference(self.columns)}."
            )

        return population_update

    def __repr__(self):
        return (
            f"PopulationView(_id={self._id}, _columns={self.columns}, _query={self._query})"
        )
