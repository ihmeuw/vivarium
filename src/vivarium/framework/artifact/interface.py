"""
==================
Artifact Interface
==================

This module provides an interface to the :class:`ArtifactManager <vivarium.framework.artifact.manager.ArtifactManager>`.


"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import pandas as pd

from vivarium.framework.artifact.manager import ArtifactManager
from vivarium.manager import Interface

if TYPE_CHECKING:
    from vivarium.types import ScalarValue


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
