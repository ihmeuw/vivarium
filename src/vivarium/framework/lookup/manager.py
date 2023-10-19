"""
=============
Lookup Tables
=============

Simulations tend to require a large quantity of data to run.  :mod:`vivarium`
provides the :class:`LookupTable` abstraction to ensure that accurate data can
be retrieved when it's needed. It's a callable object that takes in a
population index and returns data specific to the individuals represented by
that index. See the :ref:`lookup concept note <lookup_concept>` for more.

"""
from typing import TYPE_CHECKING, List, Tuple, Union

from vivarium.framework.lookup.table import LookupTable, LookupTableData
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder


class LookupTableManager(Manager):
    """Manages complex data in the simulation.

    Notes
    -----
    Client code should never access this class directly. Use ``lookup`` on the
    builder during setup to get references to LookupTable objects.

    """

    configuration_defaults = {
        "interpolation": {"order": 0, "validate": True, "extrapolate": True}
    }

    @property
    def name(self) -> str:
        return "lookup_table_manager"

    def setup(self, builder: "Builder") -> None:
        self.tables = {}
        self._pop_view_builder = builder.population.get_view
        self.clock = builder.time.clock()
        self._interpolation_order = builder.configuration.interpolation.order
        self._extrapolate = builder.configuration.interpolation.extrapolate
        self._validate = builder.configuration.interpolation.validate
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(self.build_table, allow_during=["setup"])

    def build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str]],
        parameter_columns: Union[List[str], Tuple[str]],
        value_columns: Union[List[str], Tuple[str]],
    ) -> LookupTable:
        """Construct a lookup table from input data."""
        table = self._build_table(data, key_columns, parameter_columns, value_columns)
        self._add_constraint(
            table._call, restrict_during=["initialization", "setup", "post_setup"]
        )
        return table

    def _build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str]],
        parameter_columns: Union[List[str], Tuple[str]],
        value_columns: Union[List[str], Tuple[str]],
    ) -> LookupTable:
        # We don't want to require explicit names for tables, but giving them
        # generic names is useful for introspection.
        table_number = len(self.tables)
        table = LookupTable(
            table_number,
            data,
            self._pop_view_builder,
            key_columns,
            parameter_columns,
            value_columns,
            self._interpolation_order,
            self.clock,
            self._extrapolate,
            self._validate,
        )
        self.tables[table_number] = table
        return table

    def __repr__(self) -> str:
        return "LookupTableManager()"


class LookupTableInterface:
    """The lookup table management system.

    Simulations tend to require a large quantity of data to run. ``vivarium``
    provides the :class:`Lookup Table <vivarium.framework.lookup.LookupTable>`
    abstraction to ensure that accurate data can be retrieved when it's needed.

    For more information, see :ref:`here <lookup_concept>`.

    """

    def __init__(self, manager: LookupTableManager):
        self._manager = manager

    def build_table(
        self,
        data: LookupTableData,
        key_columns: Union[List[str], Tuple[str]] = None,
        parameter_columns: Union[List[str], Tuple[str]] = None,
        value_columns: Union[List[str], Tuple[str]] = None,
    ) -> LookupTable:
        """Construct a LookupTable from input data.

        If data is a :class:`pandas.DataFrame`, an interpolation function of
        the order specified in the simulation
        :term:`configuration <Configuration>` will be calculated for each
        permutation of the set of key_columns. The columns in parameter_columns
        will be used as parameters for the interpolation functions which will
        estimate all remaining columns in the table.

        If data is a number, time, list, or tuple, a scalar table will be
        constructed with the values in data as the values in each column of
        the table, named according to value_columns.


        Parameters
        ----------
        data
            The source data which will be used to build the resulting
            :class:`LookupTable`.
        key_columns
            Columns used to select between interpolation functions. These
            should be the non-continuous variables in the data. For example
            'sex' in data about a population.
        parameter_columns
            The columns which contain the parameters to the interpolation
            functions. These should be the continuous variables. For example
            'age' in data about a population.
        value_columns
            The data columns that will be in the resulting LookupTable. Columns
            to be interpolated over if interpolation or the names of the columns
            in the scalar table.

        Returns
        -------
            LookupTable
        """
        return self._manager.build_table(data, key_columns, parameter_columns, value_columns)
