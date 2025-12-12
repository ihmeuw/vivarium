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

from typing import TYPE_CHECKING, Any, Literal, overload

import pandas as pd

import vivarium.framework.population.utilities as pop_utils
from vivarium.framework.population.exceptions import PopulationError

if TYPE_CHECKING:
    from vivarium.component import Component
    from vivarium.framework.population.manager import PopulationManager


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
        component: Component | None,
        view_id: int,
    ):
        """

        Parameters
        ----------
        manager
            The population manager for the simulation.
        component
            The component requesting this view. If None, the view will provide
            read-only access.
        view_id
            The unique identifier for this view.
        """
        self._manager = manager
        self._component = component
        self._id = view_id
        self.private_columns = component.columns_created if component else []
        self._default_query = ""

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"population_view_{self._id}"

    ###########
    # Methods #
    ###########

    def set_default_query(self, query: str) -> None:
        """Sets the default query for this population view.

        Parameters
        ----------
        query
            The new default query to apply to this population view.
        """
        self._default_query = query

    @overload
    def get_attributes(
        self,
        index: pd.Index[int],
        attributes: str,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.Series[Any]:
        ...

    @overload
    def get_attributes(
        self,
        index: pd.Index[int],
        attributes: list[str] | tuple[str, ...],
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.DataFrame:
        ...

    def get_attributes(
        self,
        index: pd.Index[int],
        attributes: str | list[str] | tuple[str, ...],
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.DataFrame | pd.Series[Any]:
        """Get a specific subset of this ``PopulationView``.

        For the rows in ``index``, return the ``attributes`` (i.e. columns) from the
        simulation's population. The resulting rows may be further filtered by the
        view's ``query`` and only return a subset of the population represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        attributes
            The columns to retrieve. If a single column is passed in via a string, the
            result will be attempted to be squeezed to a Series.
        query
            Additional conditions used to filter the index. If ``include_default_query``
            is True, it will be combined with this PopulationView's query property.
        include_default_query
            Whether to combine this view's default query with the provided ``query``.
        exclude_untracked
            Whether to exclude untracked simulants.

        Returns
        -------
            The attribute(s) requested subset to the ``index`` and ``query``. Will return
            a Series if a single column is requested or a Dataframe otherwise.

        """
        squeeze: Literal[True] | Literal[False] = isinstance(attributes, str)
        attributes = [attributes] if isinstance(attributes, str) else list(attributes)

        population = self._manager.get_population(
            attributes=attributes,
            index=index,
            query=self._build_query(query, include_default_query, exclude_untracked),
            squeeze=squeeze,
        )
        if squeeze and not isinstance(population, pd.Series):
            raise ValueError(
                "Expected a pandas Series to be returned when requesting a single "
                "attribute, but got a DataFrame instead. If you expect this attribute "
                "to be a DataFrame, you should call `get_attribute_frame()` instead."
            )
        return population

    def get_attribute_frame(
        self,
        index: pd.Index[int],
        attribute: str,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.DataFrame:
        """Get a single attribute as a DataFrame.

        For the rows in ``index``, return the ``attributes`` (i.e. columns) from the
        simulation's population. The resulting rows may be further filtered by the
        view's ``query`` and only return a subset of the population represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        attribute
            The attribute to retrieve. This attribute may contain a one or more columns.
        query
            Additional conditions used to filter the index. If ``include_default_query``
            is True, it will be combined with this PopulationView's query property.
        include_default_query
            Whether to combine this view's default query with the provided ``query``.
        exclude_untracked
            Whether to exclude untracked simulants.

        Returns
        -------
            The attribute requested subset to the ``index`` and ``query``. Will always
            return a DataFrame.

        """
        population = self._manager.get_population(
            index=index,
            attributes=[attribute],
            query=self._build_query(query, include_default_query, exclude_untracked),
        )
        if not isinstance(population, pd.DataFrame):
            raise ValueError(
                "Expected a pandas DataFrame to be returned when requesting an "
                "attribute frame, but got a Series instead. If you expect this "
                "attribute to be a Series, you should call `get_attributes()` instead."
            )
        return population

    @overload
    def get_private_columns(
        self,
        index: pd.Index[int],
        private_columns: str = ...,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.Series[Any]:
        ...

    @overload
    def get_private_columns(
        self,
        index: pd.Index[int],
        private_columns: list[str] | tuple[str, ...] = ...,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.DataFrame:
        ...

    @overload
    def get_private_columns(
        self,
        index: pd.Index[int],
        private_columns: None = None,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.Series[Any] | pd.DataFrame:
        ...

    def get_private_columns(
        self,
        index: pd.Index[int],
        private_columns: str | list[str] | tuple[str, ...] | None = None,
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.Series[Any] | pd.DataFrame:
        """Get a specific subset of this ``PopulationView's`` private columns.

        For the rows in ``index``, return the requested ``private_columns``. The
        resulting rows may be further filtered by the view's ``query`` and only
        return a subset of the data represented by the index.

        Parameters
        ----------
        index
            Index of the population to get.
        private_columns
            The private columns to retrieve. If None, all columns created by the
            component that created this view are included.
        query
            Additional conditions used to filter the index. If ``include_default_query``
            is True, it will be combined with this PopulationView's query property.
        include_default_query
            Whether to combine this view's default query with the provided ``query``.
        exclude_untracked
            Whether to exclude untracked simulants.

        Returns
        -------
            The private column(s) requested subset to the ``index`` and ``query``. Will return
            a Series if a single column is requested or a Dataframe otherwise.
        """

        if self._component is None:
            raise PopulationError(
                "This PopulationView is read-only, so it doesn't have access to get_private_columns()."
            )

        index = self.get_filtered_index(
            index,
            query=self._build_query(query, include_default_query, exclude_untracked),
            include_default_query=False,
            exclude_untracked=False,
        )

        return self._manager.get_private_columns(self._component, index, private_columns)

    def get_filtered_index(
        self,
        index: pd.Index[int],
        query: str = "",
        include_default_query: bool = True,
        exclude_untracked: bool = True,
    ) -> pd.Index[int]:
        """Get a specific index of the population.

        The requested index may be further filtered by the view's ``query``.

        Parameters
        ----------
        index
            Index of the population to get.
        query
            Additional conditions used to filter the index. If ``include_default_query``
            is True, it will be combined with this PopulationView's query property.
        include_default_query
            Whether to combine this view's default query with the provided ``query``.
        exclude_untracked
            Whether to exclude untracked simulants.

        Returns
        -------
            The requested and filtered population index.
        """

        return self.get_attributes(
            index,
            attributes=[],
            query=query,
            include_default_query=include_default_query,
            exclude_untracked=exclude_untracked,
        ).index

    def update(self, update: pd.Series[Any] | pd.DataFrame) -> None:
        """Updates the private data with the provided data.

        Parameters
        ----------
        update
            The data which should be copied into the simulation's private data. If
            the update is a :class:`pandas.DataFrame`, it can contain a subset
            of the view's columns but no extra columns. If ``pop`` is a
            :class:`pandas.Series` it must have a name that matches one of
            this view's columns unless the view only has one column in which
            case the Series will be assumed to refer to that regardless of its
            name.

        Raises
        ------
        PopulationError
            - If the ``component`` attribute is set to None (indicating that this
            view is to be read-only and thus cannot be updated).
            - If the provided update name or columns do not match columns that this
            view manages.
            - If the view is being updated with a data type inconsistent with the
            original data.
        """

        if self._component is None:
            raise PopulationError(
                "This PopulationView is read-only, so it doesn't have access to update()."
            )

        existing = pd.DataFrame(self._manager.get_private_columns(self._component))
        update_df: pd.DataFrame = self._format_update_and_check_preconditions(
            self._component.name,
            update,
            existing,
            self.private_columns,
            self._manager.creating_initial_population,
            self._manager.adding_simulants,
        )
        if self._manager.creating_initial_population:
            new_columns = list(set(update_df.columns).difference(existing.columns))
            self._manager.update(update_df[new_columns])
        elif not update_df.empty:
            update_columns = list(set(update_df.columns).intersection(existing.columns))
            updated_cols_list = []
            for column in update_columns:
                column_update = self._update_column_and_ensure_dtype(
                    update_df[column],
                    existing[column],
                    self._manager.adding_simulants,
                )
                updated_cols_list.append(column_update)
            self._manager.update(pd.concat(updated_cols_list, axis=1))

    def __repr__(self) -> str:
        name = self._component.name if self._component else "None"
        return f"PopulationView(_id={self._id}, _component={name}, private_columns={self.private_columns}, default_query={self._default_query})"

    ##################
    # Helper methods #
    ##################

    # FIXME: make this not a static method
    @staticmethod
    def _format_update_and_check_preconditions(
        component_name: str,
        update: pd.Series[Any] | pd.DataFrame,
        existing: pd.DataFrame,
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
        component_name
            The name of the component requesting the update.
        update
            The update to the private data owned by the component that created this view.
        existing
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

        update = PopulationView._coerce_to_dataframe(update, private_columns)

        unknown_simulants = len(update.index.difference(existing.index))
        if unknown_simulants:
            raise PopulationError(
                "Population updates must have an index that is a subset of the current "
                f"private data. {unknown_simulants} simulants were provided "
                "in an update with no matching index in the existing table."
            )

        if creating_initial_population:
            missing_pops = len(existing.index.difference(update.index))
            if missing_pops:
                raise PopulationError(
                    "Components must initialize all simulants during population initialization. "
                    f"Component '{component_name}' is missing updates for {missing_pops} simulants."
                )

        return update

    @staticmethod
    def _coerce_to_dataframe(
        update: pd.Series[Any] | pd.DataFrame,
        private_columns: list[str],
    ) -> pd.DataFrame:
        """Coerce all population updates to a :class:`pandas.DataFrame` format.

        Parameters
        ----------
        update
            The update to the private data owned by the component that created this view.
        private_columns
            The private column names owned by the component that created this view.

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
        if not isinstance(update, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "The population update must be a pandas Series or DataFrame. "
                f"A {type(update)} was provided."
            )

        if isinstance(update, pd.Series):
            if update.name is None:
                if len(private_columns) == 1:
                    update.name = private_columns[0]
                else:
                    raise PopulationError(
                        "Cannot update with an unnamed pandas series unless there "
                        "is only a single column in the view."
                    )

            update = pd.DataFrame(update)

        if not set(update.columns).issubset(private_columns):
            raise PopulationError(
                f"Cannot update with a DataFrame or Series that contains columns "
                f"the view does not. Dataframe contains the following extra columns: "
                f"{set(update.columns).difference(private_columns)}."
            )

        update_columns = list(update)
        if not update_columns:
            raise PopulationError(
                "The update method of population view is being called on a DataFrame "
                "with no columns."
            )

        return update

    @staticmethod
    def _update_column_and_ensure_dtype(
        update: pd.Series[Any],
        existing: pd.Series[Any],
        adding_simulants: bool,
    ) -> pd.Series[Any]:
        """Build the updated private data column with an appropriate dtype.

        This method updates any existing column values with their corresponding
        new values from the update; existing values not in the update are preserved.
        It also ensures that the resulting column has a dtype consistent with the
        original column (unless new simulants are being added).

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
        new_values = existing.array.copy()
        update_index_positional = existing.index.get_indexer(update.index)  # type: ignore [no-untyped-call]

        # Assumes the update index labels can be interpreted as an array position.
        new_values[update_index_positional] = update_values

        unmatched_dtypes = new_values.dtype != update_values.dtype
        if unmatched_dtypes and not adding_simulants:
            # This happens when the population is being grown because extending
            # the index forces columns that don't have a natural null type
            # to become 'object'
            raise PopulationError(
                "A component is corrupting the population table by modifying the dtype of "
                f"the {update.name} column from {existing.dtype} to {update.dtype}."
            )
        new_values = new_values.astype(update_values.dtype)
        new_data: pd.Series[Any] = pd.Series(
            new_values, index=existing.index, name=existing.name
        )
        return new_data

    def _build_query(
        self, query: str, include_default_query: bool, exclude_untracked: bool
    ) -> str:
        """Builds the full query for this PopulationView.

        This combines the provided query with the view's default query and the
        population manager's tracked query as appropriate.
        """
        return pop_utils.combine_queries(
            query,
            self._default_query if include_default_query else "",
            self._manager.get_tracked_query() if exclude_untracked else "",
        )
