"""
====================
The Artifact Manager
====================

This module contains the :class:`ArtifactManager`, a ``vivarium`` plugin
for handling complex data bound up in a data artifact.

"""
from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
from layered_config_tree.main import LayeredConfigTree

from vivarium.framework.artifact import ArtifactException
from vivarium.framework.artifact.artifact import Artifact
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.types import ScalarValue


class ArtifactManager(Manager):
    """The controller plugin component for managing a data artifact."""

    CONFIGURATION_DEFAULTS = {
        "input_data": {
            "artifact_path": None,
            "artifact_filter_term": None,
            "input_draw_number": None,
        }
    }

    def __init__(self) -> None:
        self._default_value_column = "value"

    @property
    def name(self) -> str:
        return "artifact_manager"

    def setup(self, builder: Builder) -> None:
        """Performs this component's simulation setup."""
        self.logger = builder.logging.get_logger(self.name)
        # because not all columns are accessible via artifact filter terms, apply config filters separately
        self.config_filter_term = validate_filter_term(
            builder.configuration.input_data.artifact_filter_term
        )
        self.artifact = self._load_artifact(builder.configuration)
        builder.lifecycle.add_constraint(self.load, allow_during=["setup"])

    def _load_artifact(self, configuration: LayeredConfigTree) -> Artifact | None:
        """Loads artifact data.

        Looks up the path to the artifact hdf file, builds a default filter,
        and generates the data artifact. Stores any configuration specified filter
        terms separately to be applied on loading, because not all columns are
        available via artifact filter terms.

        Parameters
        ----------
        configuration
            Configuration block of the model specification containing the input data parameters.

        Returns
        -------
            An interface to the data artifact.
        """
        if not configuration.input_data.artifact_path:
            return None
        artifact_path = parse_artifact_path_config(configuration)
        base_filter_terms = get_base_filter_terms(configuration)
        self.logger.info(f"Running simulation from artifact located at {artifact_path}.")
        self.logger.info(f"Artifact base filter terms are {base_filter_terms}.")
        self.logger.info(f"Artifact additional filter terms are {self.config_filter_term}.")
        return Artifact(artifact_path, base_filter_terms)

    def load(self, entity_key: str, **column_filters: int | str | Sequence[int | str]) -> Any:
        """Loads data associated with the given entity key.

        Parameters
        ----------
        entity_key
            The key associated with the expected data.
        column_filters
            Filters that subset the data by a categorical column and then remove
            the column from the raw data. They are supplied as keyword arguments
            to the load method in the form "column=value".

        Returns
        -------
            The data associated with the given key, filtered down to the
            requested subset if the data is a dataframe.
        """
        if self.artifact is None:
            raise ArtifactException("No artifact defined for simulation.")

        data = self.artifact.load(entity_key)
        if isinstance(data, pd.DataFrame):  # could be metadata dict
            data = data.reset_index()
            draw_col = [c for c in data.columns if "draw" in c]
            if draw_col:
                data = data.rename(columns={draw_col[0]: self._default_value_column})

            data = filter_data(data, self.config_filter_term, **column_filters)

        return data

    def value_columns(
        self,
    ) -> Callable[[str | pd.DataFrame | dict[str, list[ScalarValue] | list[str]]], list[str]]:
        """Returns a function that returns the value columns for the given input.

        The function can be called with either a string or a pandas DataFrame.
        If a string is provided, it is interpreted as an artifact key, and the
        value columns for the data stored at that key are returned.

        Currently, the returned function will always return ["value"].

        Returns
        -------
            A function that returns the value columns for the given input.
        """
        return lambda _: [self._default_value_column]

    def __repr__(self) -> str:
        return "ArtifactManager()"


class ArtifactInterface(Interface):
    """The builder interface for accessing a data artifact."""

    def __init__(self, manager: ArtifactManager) -> None:
        self._manager = manager

    def load(self, entity_key: str, **column_filters: int | str | Sequence[int | str]) -> Any:
        """Loads data associated with a formatted entity key.

        The provided entity key must be of the form
        {entity_type}.{measure} or {entity_type}.{entity_name}.{measure}.

        Here entity_type denotes the kind of entity being described. Examples
        include cause, risk, population, and covariates.

        The entity_name is the name of the specific entity. For example,
        if we had entity_type as cause, we might have entity_name as
        diarrheal_diseases or ischemic_heart_disease.

        Finally, measure is the name of the quantity the data describes.
        Examples of measures are incidence, disability_weight, relative_risk,
        and cost.

        Parameters
        ----------
        entity_key
            The key associated with the expected data.
        column_filters
            Filters that subset the data by a categorical column and then
            remove the column from the raw data. They are supplied as keyword
            arguments to the load method in the form "column=value".

        Returns
        -------
            The data associated with the given key filtered down to the requested subset.
        """
        return self._manager.load(entity_key, **column_filters)

    def value_columns(
        self,
    ) -> Callable[[str | pd.DataFrame | dict[str, list[ScalarValue] | list[str]]], list[str]]:
        """Returns a function that returns the value columns for the given input.

        The function can be called with either a string or a pandas DataFrame.
        If a string is provided, it is interpreted as an artifact key, and the
        value columns for the data stored at that key are returned.

        Returns
        -------
            A function that returns the value columns for the given input.
        """
        return self._manager.value_columns()

    def __repr__(self) -> str:
        return "ArtifactManagerInterface()"


def filter_data(
    data: pd.DataFrame,
    config_filter_term: str | None = None,
    **column_filters: int | str | Sequence[int | str],
) -> pd.DataFrame:
    """Uses the provided column filters and age_group conditions to subset the raw data."""
    data = _config_filter(data, config_filter_term)
    data = _subset_rows(data, **column_filters)
    data = _subset_columns(data, **column_filters)
    return data


def _config_filter(data: pd.DataFrame, config_filter_term: str | None) -> pd.DataFrame:
    if config_filter_term:
        filter_column = re.split("[<=>]", config_filter_term.split()[0])[0]
        if filter_column in data.columns:
            data = data.query(config_filter_term)
    return data


def validate_filter_term(config_filter_term: str | None) -> str | None:
    multiple_filter_indicators = [" and ", " or ", "|", "&"]
    if config_filter_term is not None and any(
        x in config_filter_term for x in multiple_filter_indicators
    ):
        raise NotImplementedError(
            "Only a single filter term via the configuration is currently supported."
        )
    return config_filter_term


def _subset_rows(
    data: pd.DataFrame, **column_filters: int | str | Sequence[int | str]
) -> pd.DataFrame:
    """Filters out unwanted rows from the data using the provided filters."""
    extra_filters = set(column_filters.keys()) - set(data.columns)
    if extra_filters:
        raise ValueError(
            f"Filtering by non-existent columns: {extra_filters}. "
            f"Available columns: {data.columns}"
        )

    for column, condition in column_filters.items():
        if isinstance(condition, (str, int)):
            condition = [condition]
        mask = pd.Series(False, index=data.index)
        for c in condition:
            mask |= data[f"{column}"] == c
        row_indexer = data[mask].index
        data = data.loc[row_indexer, :]

    return data


def _subset_columns(
    data: pd.DataFrame, **column_filters: int | str | Sequence[int | str]
) -> pd.DataFrame:
    """Filters out unwanted columns and default columns from the data using provided filters."""
    columns_to_remove = set(list(column_filters.keys()) + ["draw"])
    columns_to_remove = columns_to_remove.intersection(data.columns)
    return data.drop(columns=list(columns_to_remove))


def get_base_filter_terms(configuration: LayeredConfigTree) -> list[str]:
    """Parses default filter terms from the artifact configuration."""
    base_filter_terms = []

    draw = configuration.input_data.input_draw_number
    if draw is not None:
        base_filter_terms.append(f"draw == {draw}")

    return base_filter_terms


def parse_artifact_path_config(config: LayeredConfigTree) -> str:
    """Gets the path to the data artifact from the simulation configuration.

    The path specified in the configuration may be absolute or it may be relative
    to the location of the configuration file.

    Parameters
    ----------
    config
        The configuration block of the simulation model specification containing the artifact path.

    Returns
    -------
        The path to the data artifact.
    """
    path = Path(config.input_data.artifact_path)

    if not path.is_absolute():
        path_config = config.input_data.metadata("artifact_path")[-1]
        if path_config["source"] is None:
            raise ValueError("Insufficient information provided to find artifact.")
        path = Path(path_config["source"]).parent.joinpath(path).resolve()

    if not path.exists():
        raise FileNotFoundError(f"Cannot find artifact at path {path}")

    return str(path)
