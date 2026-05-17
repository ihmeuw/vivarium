"""
====================
Lookup Table Manager
====================

Simulations tend to require a large quantity of data to run.  :mod:`vivarium`
provides the :class:`Lookup Table <vivarium.framework.lookup.table.LookupTable>`
abstraction to ensure that accurate data can be retrieved when it's needed. It's
a callable object that takes in a population index and returns data specific to
the individuals represented by that index. See the
:ref:`lookup concept note <lookup_concept>` for more.

"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import pandas as pd
from layered_config_tree import LayeredConfigTree

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.lookup.table import DEFAULT_VALUE_COLUMN, LookupTable, is_indexed_form
from vivarium.manager import Manager
from vivarium.types import LookupTableData

VALUE_COLUMNS_DEPRECATION_MESSAGE = (
    "The `value_columns` argument to LookupTable.build_table is deprecated "
    "and will be removed in a future release when ``data`` is in indexed "
    "form. Value columns are inferred from the DataFrame columns (or the "
    "Series name) of an indexed input."
)

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.engine import Builder


class LookupTableManager(Manager):
    """Manages complex data in the simulation.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the
    builder during setup to get references to LookupTable objects.

    """

    CONFIGURATION_DEFAULTS = {
        "interpolation": {"order": 0, "validate": True, "extrapolate": True}
    }

    @property
    def name(self) -> str:
        return "lookup_table_manager"

    def __init__(self) -> None:
        super().__init__()
        self.tables: dict[str, LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]] = {}

    def setup(self, builder: Builder) -> None:
        self._logger = builder.logging.get_logger(self.name)
        self._configuration = builder.configuration
        self._get_view = builder.population.get_view
        self.clock = builder.time.clock()
        self.interpolation_order = builder.configuration.interpolation.order
        self.extrapolate = builder.configuration.interpolation.extrapolate
        self.validate_interpolation = builder.configuration.interpolation.validate
        self._add_resource = builder.resources.add_resource
        self._add_constraint = builder.lifecycle.add_constraint
        self._get_current_component = builder.components.get_current_component

        builder.lifecycle.add_constraint(
            self.build_table, allow_during=[lifecycle_states.SETUP]
        )
        builder.event.register_listener(lifecycle_states.POST_SETUP, self.on_post_setup)

    def on_post_setup(self, event: Event) -> None:
        configured_lookup_tables: dict[str, list[str]] = {}
        for config_key, config in self._configuration.items():
            if isinstance(config, LayeredConfigTree) and "data_sources" in config:
                configured_lookup_tables[config_key] = list(
                    config.get_tree("data_sources").keys()
                )

        for component_name, table_names in configured_lookup_tables.items():
            for table_name in table_names:
                full_table_name = LookupTable.get_name(component_name, table_name)
                if full_table_name not in self.tables:
                    self._logger.warning(
                        f"Component '{component_name}' configured, but didn't build lookup"
                        f" table '{table_name}' during setup."
                    )

    def build_table(
        self,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str | None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        """Construct a lookup table from input data."""
        if value_columns is not None and is_indexed_form(data):
            warnings.warn(
                VALUE_COLUMNS_DEPRECATION_MESSAGE,
                DeprecationWarning,
                stacklevel=3,
            )
        component = self._get_current_component()
        table = self._build_table(component, data, name, value_columns)
        self._add_resource(table)
        self._add_constraint(
            table._call,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )
        self._add_constraint(
            table.set_data,
            restrict_during=[lifecycle_states.POPULATION_CREATION],
        )
        return table

    def _build_table(
        self,
        component: Component,
        data: LookupTableData,
        name: str,
        value_columns: list[str] | tuple[str, ...] | str | None,
    ) -> LookupTable[pd.Series[Any]] | LookupTable[pd.DataFrame]:
        # We don't want to require explicit names for tables, but giving them
        # generic names is useful for introspection.
        if not name:
            name = f"lookup_table_{len(self.tables)}"

        table = LookupTable(
            name=name,
            component=component,
            data=data,
            value_columns=value_columns or DEFAULT_VALUE_COLUMN,
            manager=self,
            population_view=self._get_view(),
        )

        self.tables[table.name] = table

        return table

    def __repr__(self) -> str:
        return "LookupTableManager()"
