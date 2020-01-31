"""
=============
HDF Interface
=============

A convenience wrapper around the :mod:`tables` and :mod:`pandas` HDF
interfaces.

Public Interface
----------------

The public interface consists of 5 functions:

.. list-table:: HDF Public Interface
   :widths: 20 60
   :header-rows: 1

   * - Function
     - Description
   * - :func:`touch`
     - Creates an HDF file, wiping an existing file if necessary.
   * - :func:`write`
     - Stores data at a key in an HDF file.
   * - :func:`load`
     - Loads (potentially filtered) data from a key in an HDF file.
   * - :func:`remove`
     - Clears data from a key in an HDF file.
   * - :func:`get_keys`
     - Gets all available HDF keys from an HDF file.

Contracts
+++++++++

- All functions in the public interface accept both :class:`pathlib.Path` and
  normal Python :class:`str` objects for paths.
- All functions in the public interface accept only :class:`str` objects
  as representations of the keys in the hdf file.  The strings must be
  formatted as ``"type.name.measure"`` or ``"type.measure"``.

"""
import json
from pathlib import Path
from typing import Any, List, Optional, Union
import re

import pandas as pd
import tables
from tables.nodes import filenode

PandasObj = (pd.DataFrame, pd.Series)

####################
# Public interface #
####################


def touch(path: Union[str, Path]):
    """Creates an HDF file, wiping an existing file if necessary.

    If the given path is proper to create a HDF file, it creates a new
    HDF file.

    Parameters
    ----------
    path
        The path to the HDF file.

    Raises
    ------
    ValueError
        If the non-proper path is given to create a HDF file.

    """
    path = _get_valid_hdf_path(path)

    with tables.open_file(str(path), mode='w'):
        pass


def write(path: Union[str, Path], entity_key: str, data: Any):
    """Writes data to the HDF file at the given path to the given key.

    Parameters
    ----------
    path
        The path to the HDF file to write to.
    entity_key
        A string representation of the internal HDF path where we want to
        write the data. The key must be formatted as ``"type.name.measure"``
        or ``"type.measure"``.
    data
        The data to write. If it is a :mod:`pandas` object, it will be
        written using a :class:`pandas.HDFStore` or :func:`pandas.to_hdf`.
        If it is some other kind of python object, it will first be encoded
        as json with :func:`json.dumps` and then written to the provided
        key.

    Raises
    ------
    ValueError
        If the path or entity_key are improperly formatted.

    """
    path = _get_valid_hdf_path(path)
    entity_key = EntityKey(entity_key)

    if isinstance(data, PandasObj):
        _write_pandas_data(path, entity_key, data)
    else:
        _write_json_blob(path, entity_key, data)


def load(path: Union[str, Path], entity_key: str, filter_terms: Optional[List[str]],
         column_filters: Optional[List[str]]) -> Any:
    """Loads data from an HDF file.

    Parameters
    ----------
    path
        The path to the HDF file to load the data from.
    entity_key
        A representation of the internal HDF path where the data is located.
    filter_terms
        An optional list of terms used to filter the rows in the data.
        The terms must be formatted in a way that is suitable for use with
        the ``where`` argument of :func:`pd.read_hdf`. Only
        filters applying to existing columns in the data are used.
    column_filters
        An optional list of columns to load from the data.

    Raises
    ------
    ValueError
        If the path or entity_key are improperly formatted.

    Returns
    -------
        The data stored at the the given key in the HDF file.

    """
    path = _get_valid_hdf_path(path)
    entity_key = EntityKey(entity_key)

    with tables.open_file(str(path)) as file:
        node = file.get_node(entity_key.path)
        if isinstance(node, tables.earray.EArray):
            # This should be a json encoded document rather than a pandas dataframe
            with filenode.open_node(node) as file_node:
                data = json.load(file_node)
        else:
            filter_terms = _get_valid_filter_terms(filter_terms, node.table.colnames)
            with pd.HDFStore(str(path), complevel=9, mode='r') as store:
                metadata = store.get_storer(entity_key.path).attrs.metadata  # NOTE: must use attrs. write this up

            if metadata.get('is_empty', False):
                data = pd.read_hdf(path, entity_key.path, where=filter_terms)
                data = data.set_index(list(data.columns))  # undoing transform performed on write
            else:
                data = pd.read_hdf(path, entity_key.path, where=filter_terms, columns=column_filters)

    return data


def remove(path: Union[str, Path], entity_key: str):
    """Removes a piece of data from an HDF file.

    Parameters
    ----------
    path :
        The path to the HDF file to remove the data from.
    entity_key :
        A representation of the internal HDF path where the data is located.

    Raises
    ------
    ValueError
        If the path or entity_key are improperly formatted.

    """
    path = _get_valid_hdf_path(path)
    entity_key = EntityKey(entity_key)

    with tables.open_file(str(path), mode='a') as file:
        file.remove_node(entity_key.path, recursive=True)


def get_keys(path: str) -> List[str]:
    """Gets key representation of all paths in an HDF file.

    Parameters
    ----------
    path :
        The path to the HDF file.

    Returns
    -------
        A list of key representations of the internal paths in the HDF.
    """
    path = _get_valid_hdf_path(path)
    with tables.open_file(str(path)) as file:
        keys = _get_keys(file.root)
    return keys


class EntityKey(str):
    """A convenience wrapper that translates artifact keys.

    This class provides several representations of the artifact keys that
    are useful when working with the :mod:`pandas` and :mod:`tables`
    HDF interfaces.

    """

    def __init__(self, key):
        """
        Parameters
        ----------
        key
            The string representation of the entity key.  Must be formatted
            as ``"type.name.measure"`` or ``"type.measure"``.
        """
        elements = [e for e in key.split('.') if e]
        if len(elements) not in [2, 3] or len(key.split('.')) != len(elements):
            raise ValueError(f'Invalid format for HDF key: {key}. '
                             'Acceptable formats are "type.name.measure" and "type.measure"')
        super().__init__()

    @property
    def type(self) -> str:
        """The type of the entity represented by the key."""
        return self.split('.')[0]

    @property
    def name(self) -> str:
        """The name of the entity represented by the key"""
        return self.split('.')[1] if len(self.split('.')) == 3 else ''

    @property
    def measure(self) -> str:
        """The measure associated with the data represented by the key."""
        return self.split('.')[-1]

    @property
    def group_prefix(self) -> str:
        """The HDF group prefix for the key."""
        return '/'+self.type if self.name else '/'

    @property
    def group_name(self) -> str:
        """The HDF group name for the key."""
        return self.name if self.name else self.type

    @property
    def group(self) -> str:
        """The full path to the group for this key."""
        return self.group_prefix + '/' + self.group_name if self.name else self.group_prefix + self.group_name

    @property
    def path(self) -> str:
        """The full HDF path associated with this key."""
        return self.group + '/' + self.measure

    def with_measure(self, measure: str) -> 'EntityKey':
        """Replaces this key's measure with the provided one.

        Parameters
        ----------
        measure :
            The measure to replace this key's measure with.

        Returns
        -------
            A new EntityKey with the updated measure.
        """
        if self.name:
            return EntityKey(f'{self.type}.{self.name}.{measure}')
        else:
            return EntityKey(f'{self.type}.{measure}')

    def __eq__(self, other: 'EntityKey') -> bool:
        return isinstance(other, str) and str(self) == str(other)

    def __ne__(self, other: 'EntityKey') -> bool:
        return not self == other

    def __hash__(self):
        return hash(str(self))

    def __repr__(self) -> str:
        return f'EntityKey({str(self)})'


#####################
# Private utilities #
#####################


def _get_valid_hdf_path(path: Union[str, Path]) -> Path:
    valid_suffixes = ['.hdf', '.h5']

    path = Path(path)
    if path.suffix not in valid_suffixes:
        raise ValueError(f'{str(path)} has an invalid HDF suffix {path.suffix}.'
                         f' HDF files must have one of {valid_suffixes} as a path suffix.')
    return path


def _write_pandas_data(path: Path, entity_key: EntityKey, data: Union[PandasObj]):
    """Write data in a pandas format to an HDF file.

    This method currently supports :class:`pandas DataFrame` objects, with or
    with or without columns, and :class:`pandas.Series` objects.

    """
    if data.empty:
        # Our data is indexed, sometimes with no other columns. This leaves an
        # empty dataframe that store.put will silently fail to write in table
        # format.
        data = data.reset_index()
        if data.empty:
            raise ValueError("Cannot write an empty dataframe that does not have an index.")
        metadata = {'is_empty': True}
        data_columns = True
    else:
        metadata = {'is_empty': False}
        data_columns = None

    with pd.HDFStore(str(path), complevel=9) as store:
        store.put(entity_key.path, data, format="table", data_columns=data_columns)
        store.get_storer(entity_key.path).attrs.metadata = metadata  # NOTE: must use attrs. write this up


def _write_json_blob(path: Path, entity_key: EntityKey, data: Any):
    """Writes a Python object as json to the HDF file at the given path."""
    with tables.open_file(str(path), "a") as store:

        if entity_key.group_prefix not in store:
            store.create_group('/', entity_key.type)

        if entity_key.group not in store:
            store.create_group(entity_key.group_prefix, entity_key.group_name)

        with filenode.new_node(store, where=entity_key.group, name=entity_key.measure) as fnode:
            fnode.write(bytes(json.dumps(data), "utf-8"))


def _get_keys(root: tables.node.Node, prefix: str = '') -> List[str]:
    """Recursively formats the paths in an HDF file into a key format."""
    keys = []
    for child in root:
        child_name = _get_node_name(child)
        if isinstance(child, tables.earray.EArray):  # This is the last node
            keys.append(f'{prefix}.{child_name}')
        elif isinstance(child, tables.table.Table):  # Parent was the last node
            keys.append(prefix)
        else:
            new_prefix = f'{prefix}.{child_name}' if prefix else child_name
            keys.extend(_get_keys(child, new_prefix))

    # Clean up some weird meta groups that get written with dataframes.
    keys = [k for k in keys if '.meta.' not in k]
    return keys


def _get_node_name(node: tables.node.Node) -> str:
    """Gets the name of a node from its string representation."""
    node_string = str(node)
    node_path = node_string.split()[0]
    node_name = node_path.split('/')[-1]
    return node_name


def _get_valid_filter_terms(filter_terms, colnames):
    """Removes any filter terms referencing non-existent columns

    Parameters
    ----------
    filter_terms
        A list of terms formatted so as to be used in the `where` argument of
        :func:`pd.read_hdf`.
    colnames :
        A list of column names present in the data that will be filtered.

    Returns
    -------
        The list of valid filter terms (terms that do not reference any column
        not existing in the data). Returns none if the list is empty because
        the `where` argument doesn't like empty lists.

    """
    if not filter_terms:
        return None
    valid_terms = filter_terms.copy()
    for term in filter_terms:
        # first strip out all the parentheses - the where in read_hdf
        # requires all references to be valid
        t = re.sub('[()]', '', term)
        # then split each condition out
        t = re.split('[&|]', t)
        # get the unique columns referenced by this term
        term_columns = set([re.split('[<=>\s]', i.strip())[0] for i in t])
        if not term_columns.issubset(colnames):
            valid_terms.remove(term)
    return valid_terms if valid_terms else None
