"""
===================
The Population View
===================

The :class:`PopulationView` is a user-facing abstraction that manages read and write 
access to the underlying :term:`population state table <Population State Table>`.
It has two primary responsibilities:

    1. To provide user access to subsets of the state table when it is safe to do so.
    2. To allow the user to update private data in a controlled way.

"""
from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal, overload

import pandas as pd

import vivarium.framework.population.utilities as pop_utils
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.population.exceptions import PopulationError

if TYPE_CHECKING:
    from vivarium.component import Component
    from vivarium.framework.population.manager import PopulationManager


class PopulationView:
    """A read/write manager for the population state table.

    It can be used to both read and update the state of the population. While a
    PopulationView can read any column, it can only write those columns that the
    component it is attached to created (i.e. that component's private columns).

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

    ##############
    # Properties #
    ##############

    @property
    def name(self) -> str:
        return f"population_view_{self._id}"

    @property
    def private_columns(self) -> list[str]:
        """The names of private columns managed by this PopulationView.

        These private columns are those that were created by the component
        that created this view.
        """
        if self._component is None:
            raise PopulationError(
                "This PopulationView is read-only, so it doesn't have access to private_columns."
            )
        return self._manager.get_private_column_names(self._component.name)

    ###########
    # Methods #
    ###########

    @overload
    def get(
        self,
        index: pd.Index[int],
        attributes: str,
        query: str = "",
        include_untracked: bool | None = None,
        skip_post_processor: Literal[False] = False,
        mode: Literal["default"] = "default",
    ) -> pd.Series[Any]:
        ...

    @overload
    def get(
        self,
        index: pd.Index[int],
        attributes: list[str] | tuple[str, ...],
        query: str = "",
        include_untracked: bool | None = None,
        skip_post_processor: Literal[False] = False,
        mode: Literal["default"] = "default",
    ) -> pd.DataFrame:
        ...

    @overload
    def get(
        self,
        index: pd.Index[int],
        attributes: str | list[str] | tuple[str, ...],
        query: str = "",
        include_untracked: bool | None = None,
        skip_post_processor: Literal[True] = ...,
        mode: Literal["default", "source", "no-post-processors"] = "default",
    ) -> Any:
        ...

    @overload
    def get(
        self,
        index: pd.Index[int],
        attributes: str | list[str] | tuple[str, ...],
        query: str = "",
        include_untracked: bool | None = None,
        skip_post_processor: Literal[False] = False,
        mode: Literal["source", "no-post-processors"] = ...,
    ) -> Any:
        ...

    def get(
        self,
        index: pd.Index[int],
        attributes: str | list[str] | tuple[str, ...],
        query: str = "",
        include_untracked: bool | None = None,
        skip_post_processor: Literal[True, False] = False,
        mode: Literal["default", "source", "no-post-processors"] = "default",
    ) -> Any:
        """Gets a specific subset of the population state table.

        For the rows in ``index``, return the ``attributes`` (i.e. columns) from the
        state table. The resulting rows may be further filtered by the call's ``query``
        and whether or not to include untracked simulants.

        Parameters
        ----------
        index
            Index of the population to get. This may be further filtered by various
            query conditions.
        attributes
            The attributes to retrieve. If a single attribute is passed in via a
            string, the result will be squeezed to a Series if possible.
        query
            Additional conditions used to filter the index.
        include_untracked
            Whether to include untracked simulants. If None (default), untracked
            simulants are excluded unless this pipeline was called during population
            creation or inside another pipeline call. Untracked simulants are always
            included if True and always excluded if False.
        skip_post_processor
            Whether we should invoke the post-processor on the combined
            source and mutator output or return without post-processing.
            This is useful when the post-processor acts as some sort of final
            unit conversion (e.g. the rescale post processor).
        mode
            The mode for pipeline evaluation. One of "default", "source",
            or "no-post-processors".

        Notes
        -----
        If ``skip_post_processor`` is True, the returned data will not be squeezed.

        Returns
        -------
            The attribute(s) requested subset to the ``index`` and filtered using
            the various optional queries. If ``skip_post_processor`` is False, will
            return a Series if a single attribute is requested or a Dataframe otherwise.

        Raises
        ------
        ValueError
            If the result is expected to be a Series but is not.
            If an invalid mode is provided.
        """
        valid_modes = ("default", "source", "no-post-processors")
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {valid_modes}.")

        # Translate skip_post_processor into mode
        if skip_post_processor:
            warnings.warn(
                "The 'skip_post_processor' parameter is deprecated. "
                "Use mode='no-post-processors' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            mode = "no-post-processors"

        squeeze: Literal[True, False] = isinstance(attributes, str)
        attributes = [attributes] if isinstance(attributes, str) else list(attributes)

        population = self._manager.get_population(
            attributes=attributes,
            index=index,
            query=self._build_query(query, include_untracked),
            squeeze=squeeze,
            mode=mode,
        )
        if mode == "default" and squeeze and not isinstance(population, pd.Series):
            raise ValueError(
                "Expected a pandas Series to be returned when requesting a single "
                "attribute, but got a DataFrame instead. If you expect this attribute "
                "to be a DataFrame, you should call `get_frame()` instead."
            )
        return population

    def get_frame(
        self,
        index: pd.Index[int],
        attribute: str,
        query: str = "",
        include_untracked: bool | None = None,
    ) -> pd.DataFrame:
        """Gets a single attribute as a DataFrame.

        For the rows in ``index``, return the ``attributes`` (i.e. columns) from the
        state table. The resulting rows may be further filtered by the call's ``query``
        and whether or not to include untracked simulants.

        Parameters
        ----------
        index
            Index of the population to get.
        attribute
            The attribute to retrieve. This attribute may contain one or more columns.
        query
            Additional conditions used to filter the index.
        include_untracked
            Whether to include untracked simulants. If None (default), untracked
            simulants are excluded unless this pipeline was called during population
            creation or inside another pipeline call. Untracked simulants are always
            included if True and always excluded if False.

        Notes
        -----
        The difference between this method and ``get`` is subtle. This
        method always returns a dataframe even if the requested attribute contains
        a single column. Further, in the event the attribute has multi-level columns,
        it will be squeezed to only return the inner columns.

        Calling ``get`` to request a list of a single attribute seems
        identical to this, but in that case the underlying data would not be squeezed
        at all, i.e. a dataframe with multi-level columns would also return the
        outer columns.

        Returns
        -------
            The attribute requested subset to the ``index`` and filtered using
            the various optional queries. Will always return a DataFrame.

        """
        return pd.DataFrame(
            self._manager.get_population(
                index=index,
                attributes=[attribute],
                query=self._build_query(query, include_untracked),
            )
        )

    def get_filtered_index(
        self,
        index: pd.Index[int],
        query: str = "",
        include_untracked: bool | None = None,
    ) -> pd.Index[int]:
        """Gets a specific index of the population.

        The requested index may be further filtered by the call's ``query`` and
        whether or not to include untracked simulants.

        Parameters
        ----------
        index
            Index of the population to get.
        query
            Additional conditions used to filter the index.
        include_untracked
            Whether to include untracked simulants. If None (default), untracked
            simulants are excluded unless this pipeline was called during population
            creation or inside another pipeline call. Untracked simulants are always
            included if True and always excluded if False.

        Returns
        -------
            The requested and filtered population index.
        """

        return self.get(
            index,
            attributes=[],
            query=query,
            include_untracked=include_untracked,
        ).index

    def initialize(self, data: pd.Series[Any] | pd.DataFrame) -> None:
        """Initialize private columns with the provided data.

        Use this method during simulant initialization (both initial and when adding
        new simulants) to set the initial values of private columns. Column names
        are inferred from the data (Series name or DataFrame columns).

        Parameters
        ----------
        data
            The initial values for private columns. If a :class:`pandas.Series`,
            its ``name`` identifies the column. If a :class:`pandas.DataFrame`,
            its column names identify the columns.

        Raises
        ------
        PopulationError
            - If this view is read-only.
            - If called outside of simulant initialization.
            - If the data contains columns not managed by this view.
            - If the data has simulants not in the population.
            - If the data is missing simulants during initial population creation.
        TypeError
            If the data is not a Series or DataFrame.
        """
        if self._component is None:
            raise PopulationError(
                "This PopulationView is read-only, so it doesn't have access to initialize()."
            )
        if not self._manager.adding_simulants:
            raise PopulationError(
                "initialize() can only be called during simulant initialization. "
                "Use update() to modify existing data."
            )

        data_df = self._coerce_init_data(data, self.private_columns)
        existing = pd.DataFrame(self._manager.get_private_columns(self._component))

        unknown_simulants = len(data_df.index.difference(existing.index))
        if unknown_simulants:
            raise PopulationError(
                "Population updates must have an index that is a subset of the current "
                f"private data. {unknown_simulants} simulants were provided "
                "in an update with no matching index in the existing table."
            )

        if self._manager.creating_initial_population:
            missing_pops = len(existing.index.difference(data_df.index))
            if missing_pops:
                raise PopulationError(
                    "Components must initialize all simulants during population "
                    f"initialization. Component '{self._component.name}' is missing "
                    f"updates for {missing_pops} simulants."
                )
            new_columns = list(set(data_df.columns).difference(existing.columns))
            self._manager.update(data_df[new_columns])
        elif not data_df.empty:
            update_columns = list(set(data_df.columns).intersection(existing.columns))
            updated_cols_list = []
            for column in update_columns:
                column_update = self._update_column_and_ensure_dtype(
                    data_df[column],
                    existing[column],
                    adding_simulants=True,
                )
                updated_cols_list.append(column_update)
            self._manager.update(pd.concat(updated_cols_list, axis=1))

    @overload
    def update(
        self,
        columns: str,
        modifier: Callable[[pd.Series[Any]], pd.Series[Any]],
    ) -> None:
        ...

    @overload
    def update(
        self,
        columns: list[str],
        modifier: Callable[[pd.DataFrame], pd.DataFrame],
    ) -> None:
        ...

    def update(
        self,
        columns: str | list[str],
        modifier: Callable[..., pd.Series[Any] | pd.DataFrame],
    ) -> None:
        """Update private columns by applying a modifier to the current data.

        Read the current values of the specified private columns, pass them to
        ``modifier``, and write the result back. The modifier receives a
        :class:`pandas.Series` when ``columns`` is a string or a
        :class:`pandas.DataFrame` when ``columns`` is a list. It should return
        data in the same form, optionally with a subset of the original index
        (in which case only those rows are updated).

        Parameters
        ----------
        columns
            The private column(s) to update. A string for a single column
            or a list of strings for multiple columns.
        modifier
            A callable that takes the current column data and returns the
            updated values. May return a subset of the original index to
            update only some rows.

        Raises
        ------
        PopulationError
            - If this view is read-only.
            - If the modifier returns data with unexpected columns or simulants.
        TypeError
            If the modifier does not return a Series, DataFrame, or scalar.
        """
        if self._component is None:
            raise PopulationError(
                "This PopulationView is read-only, so it doesn't have access to update()."
            )

        if isinstance(columns, str):
            squeeze = True
            column_list = [columns]
        else:
            squeeze = False
            column_list = list(columns)

        current_data = self._manager.get_private_columns(self._component, columns=columns)
        result = modifier(current_data.copy())
        result_df = self._coerce_update_result(result, column_list, current_data.index)

        if not result_df.empty:
            existing_full = pd.DataFrame(current_data) if squeeze else current_data
            updated_cols_list = []
            for column in result_df.columns:
                column_update = self._update_column_and_ensure_dtype(
                    result_df[column],
                    existing_full[column],
                    adding_simulants=self._manager.adding_simulants,
                )
                updated_cols_list.append(column_update)
            self._manager.update(pd.concat(updated_cols_list, axis=1))

    def __repr__(self) -> str:
        name = self._component.name if self._component else "None"
        private_columns = self.private_columns if self._component else "N/A"
        return f"PopulationView(_id={self._id}, _component={name}, private_columns={private_columns})"

    ##################
    # Helper methods #
    ##################

    @staticmethod
    def _coerce_update_result(
        result: Any,
        columns: list[str],
        existing_index: pd.Index[int],
    ) -> pd.DataFrame:
        """Coerce the return value of a modifier callable to a DataFrame.

        Parameters
        ----------
        result
            The return value of the modifier callable.
        columns
            The column names that were passed to the modifier.
        existing_index
            The index of all simulants in the private data.

        Returns
        -------
            The result coerced to a DataFrame.

        Raises
        ------
        PopulationError
            If the result contains unexpected columns or simulants.
        TypeError
            If the result is not a Series, DataFrame, or scalar.
        """
        if result is None:
            raise TypeError("The modifier returned None. Did you forget a return statement?")

        if isinstance(result, pd.DataFrame):
            coerced = result
        elif isinstance(result, pd.Series):
            if result.name is None:
                if len(columns) == 1:
                    result = result.rename(columns[0])
                else:
                    raise PopulationError(
                        "The modifier returned an unnamed Series, but multiple columns "
                        "were requested. The Series must be named to identify which "
                        "column it corresponds to, or return a DataFrame instead."
                    )
            coerced = pd.DataFrame(result)
        else:
            try:
                coerced = pd.DataFrame({col: result for col in columns}, index=existing_index)
            except (ValueError, TypeError):
                raise TypeError(
                    "The modifier must return a pandas Series, DataFrame, or scalar. "
                    f"Got {type(result)}."
                )

        extra_cols = set(coerced.columns).difference(columns)
        if extra_cols:
            raise PopulationError(
                f"The modifier returned data with unexpected columns: {extra_cols}."
            )

        missing_cols = set(columns).difference(coerced.columns)
        if missing_cols:
            raise PopulationError(
                f"The modifier did not return data for all requested columns. "
                f"Missing: {missing_cols}."
            )

        unknown = coerced.index.difference(existing_index)
        if len(unknown):
            raise PopulationError(
                f"The modifier returned {len(unknown)} simulants not in the population."
            )

        return coerced

    @staticmethod
    def _coerce_init_data(
        update: pd.Series[Any] | pd.DataFrame,
        private_columns: list[str],
    ) -> pd.DataFrame:
        """Coerces all population updates to a :class:`pandas.DataFrame` format.

        Parameters
        ----------
        update
            The update to the private data owned by the component that created this view.
        private_columns
            The private column names owned by the component that created this view.

        Returns
        -------
            The input data formatted as a DataFrame.
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
        """Builds the updated private column with an appropriate dtype.

        This method updates any existing private column values with their corresponding
        new values from the update; existing values not in the update are preserved.
        It also ensures that the resulting column has a dtype consistent with the
        original column (unless new simulants are being added).

        Parameters
        ----------
        update
            The new column values for a subset of the existing index.
        existing
            The existing column values for all simulants.
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

    def _build_query(self, query: str, include_untracked: bool | None) -> str:
        """Builds the full query for this PopulationView.

        This combines the provided query with the population manager's tracked query
        as appropriate.

        Parameters
        ----------
        query
            An explicit query string to filter the index.
        include_untracked
            Controls whether the tracked query is applied:

            - None (default): The tracked query is applied at top level, but automatically
              suppressed during nested pipeline evaluation (``pipeline_evaluation_depth > 0``)
              or during initialization population creation lifecycle phases.
            - True: The tracked query is always suppressed (untracked simulants are included).
            - False: The tracked query is always applied (untracked simulants are excluded).

        Notes
        -----
        Only the tracked query is affected. Any explicit ``query`` argument is
        always preserved so that pipeline sources can further subdivide the index.
        """
        skip_tracked_query = include_untracked is True or (
            include_untracked is None
            and (
                self._manager.get_current_state() == lifecycle_states.POPULATION_CREATION
                or self._manager.pipeline_evaluation_depth > 0
            )
        )
        return pop_utils.combine_queries(
            query,
            self._manager.get_tracked_query() if not skip_tracked_query else "",
        )
