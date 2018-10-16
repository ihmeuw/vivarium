import warnings

import pandas as pd
import numpy as np
from scipy import interpolate
from typing import Union, List, Tuple, TypeVar

ParameterType = TypeVar('ParameterType', List[List[str]], List[Tuple[str, str, str]])


class Interpolation:
    """A callable that returns the result of an interpolation function over input data.

        Attributes
        ----------
        data :
            The data from which to build the interpolation. Contains
            cateogrical_parameters and continuous_parameters.
        categorical_parameters :
            Column names to be used as categorical parameters in Interpolation
            to select between interpolation functions.
        continuous_parameters :
            Column names to be used as continuous parameters in Interpolation. If
            bin edges, should be of the form (column name used in call, column name
            for left bin edge, column name for right bin edge). If order is 0 and
            any continuous parameter is given without bin edges, continuous bins
            are created using the given parameter as midpoints.
        order :
            Order of interpolation.
        """

    def __init__(self, data: pd.DataFrame, categorical_parameters: Union[List[str], Tuple[str]],
                 continuous_parameters: ParameterType, order: int, extrapolate: bool):
        # TODO: allow for order 1 interpolation with binned edges
        if order != 0:
            raise NotImplementedError(f'Interpolation is only supported for order 0. You specified order {order}')

        self.key_columns = categorical_parameters
        self.parameter_columns, self._data, self.value_columns = validate_parameters(data, categorical_parameters,
                                                                                     continuous_parameters, order)
        self.order = order
        self.extrapolate = extrapolate

        if self.key_columns:
            # Since there are key_columns we need to group the table by those
            # columns to get the sub-tables to fit
            sub_tables = self._data.groupby(list(self.key_columns))
        else:
            # There are no key columns so we will fit the whole table
            sub_tables = {None: self._data}.items()

        self.interpolations = {}

        for key, base_table in sub_tables:
            if base_table.empty:    # if one of the key columns is a category and not all values are present in data
                continue
            # if order 0, we can interpolate all values at once
            if order == 0:
                self.interpolations[key] = Order0Interp(base_table, self.parameter_columns,
                                                        self.value_columns, self.extrapolate)
            else:
                # For each permutation of the key columns build interpolations
                self.interpolations[key] = {}

                for value_column in self.value_columns:
                    # For each value in the table build an interpolation function
                    if len(self.parameter_columns) == 2:
                        # 2 variable interpolation
                        index, column = self.parameter_columns
                        table = base_table.pivot(index=index, columns=column, values=value_column)
                        x = table.index.values
                        y = table.columns.values
                        z = table.values
                        func = interpolate.RectBivariateSpline(x=x, y=y, z=z, ky=order, kx=order).ev
                    else:
                        # 1 variable interpolation
                        base_table = base_table.sort_values(by=self.parameter_columns[0])
                        x = base_table[self.parameter_columns[0]]
                        y = base_table[value_column]
                        func = interpolate.InterpolatedUnivariateSpline(x, y, k=order)
                    self.interpolations[key][value_column] = func

    def __call__(self, interpolants: pd.DataFrame) -> pd.DataFrame:
        """Get the interpolated results for the parameters in interpolants.

        Parameters
         ----------
        interpolants :
            Data frame containing the parameters to interpolate..

        Returns
        -------
        pd.DataFrame
            A table with the interpolated values for the given interpolants.
        """

        validate_call_data(interpolants, self.key_columns, self.parameter_columns)

        if self.key_columns:
            sub_tables = interpolants.groupby(list(self.key_columns))
        else:
            sub_tables = [(None, interpolants)]
        # specify some numeric type for columns so they won't be objects but will updated with whatever
        # column type actually is
        result = pd.DataFrame(index=interpolants.index, columns=self.value_columns, dtype=np.float64)
        for key, sub_table in sub_tables:
            if sub_table.empty:
                continue
            if self.order == 0:  # we can interpolate all value columns at once
                df = self.interpolations[key](sub_table)
                result.loc[sub_table.index, self.value_columns] = df.loc[sub_table.index, self.value_columns]
            else:
                funcs = self.interpolations[key]
                parameters = tuple(sub_table[k] for k in self.parameter_columns)
                for value_column, func in funcs.items():
                    out = func(*parameters)
                    # This reshape is necessary because RectBivariateSpline and InterpolatedUnivariateSpline return results
                    # in slightly different shapes and we need them to be consistent
                    if out.shape:
                        result.loc[sub_table.index, value_column] = out.reshape((out.shape[0],))
                    else:
                        result.loc[sub_table.index, value_column] = out

        return result

    def __repr__(self):
        return "Interpolation()"


def validate_parameters(data, categorical_parameters, continuous_parameters, order):
    if order not in [0, 1]:
        raise ValueError('Only order 0 and order 1 interpolations are supported. '
                         f'You specified {order}')

    if len(continuous_parameters) not in [1, 2] and order != 0:
        raise ValueError("Interpolation over more than two parameters is only supported"
                         "for order 0. For all other orders, only interpolation over 1 or "
                         "2 variables is supported.")

    if len(continuous_parameters) < 1:
        raise ValueError("You must supply at least one continuous parameter over which to interpolate.")

    out = []
    for p in continuous_parameters:
        if not isinstance(p, (List, Tuple)) or len(p) != 3:
            raise ValueError(f'Interpolation is only supported for binned data. You must specify a list or tuple '
                             f'containing, in order, the column name used when interpolation is called, '
                             f'the column name for the left edge (inclusive), and the column name for '
                             f'the right edge (exclusive). You provided {p}.')

        if p[0] in data: # if for some reason, we have the call column drop so it doesn't get interpolated
            data = data.drop(p[0], axis='columns')

        if len(data[p[1]].unique()) > order:
            out.append(p)
        else:
            warnings.warn(f"You requested an order {order} interpolation over the parameter {p[1:]}, "
                          f"however there are only {len(data[p[1]].unique())} unique values for {p[1:]}"
                          f"which is insufficient to support the requested interpolation order."
                          f"The parameter will be dropped from the interpolation.")
            data = data.drop(p[1], axis='columns')
            data = data.drop(p[2], axis='columns')

    # break out the individual columns from binned column name lists
    param_cols = [col for p in continuous_parameters for col in p]

    # These are the columns which the interpolation function will approximate
    value_columns = sorted(data.columns.difference(set(categorical_parameters) | set(param_cols)))
    if not value_columns:
        raise ValueError(f"No non-parameter data. Available columns: {data.columns}, "
                         f"Parameter columns: {set(categorical_parameters)|set(continuous_parameters)}")
    return out, data, value_columns


def validate_call_data(data, key_columns, parameter_columns):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Interpolations can only be called on pandas.DataFrames. You'
                        f'passed {type(data)}.')
    callable_param_cols = [p[0] for p in parameter_columns]

    if not set(callable_param_cols) <= set(data.columns.values.tolist()):
        raise ValueError(f'The continuous parameter columns with which you built the Interpolation must all '
                         f'be present in the data you call it on. The Interpolation has key '
                         f'columns: {callable_param_cols} and your data has columns: '
                         f'{data.columns.values.tolist()}')

    if key_columns and not set(key_columns) <= set(data.columns.values.tolist()):
        raise ValueError(f'The key (categorical) columns with which you built the Interpolation must all'
                         f'be present in the data you call it on. The Interpolation has key'
                         f'columns: {key_columns} and your data has columns: '
                         f'{data.columns.values.tolist()}')


def check_data_complete(data, parameter_columns):
    """ For any parameters specified with edges, make sure edges
    don't overlap and don't have any gaps. Assumes that edges are
    specified with ends and starts overlapping (but one exclusive and
    the other inclusive) so can check that end of previous == start
    of current.

    If multiple parameters, make sure all combinations of parameters
    are present in data."""

    param_edges = [p[1:] for p in parameter_columns if isinstance(p, (Tuple, List))]  # strip out call column name

    # check no overlaps/gaps
    for p in param_edges:
        other_params = [p_ed[0] for p_ed in param_edges if p_ed != p]
        if other_params:
            sub_tables = data.groupby(list(other_params))
        else:
            sub_tables = {None: data}.items()

        n_p_total = len(set(data[p[0]]))

        for _, table in sub_tables:

            param_data = table[[p[0], p[1]]].copy().sort_values(by=p[0])
            start, end = param_data[p[0]].reset_index(drop=True), param_data[p[1]].reset_index(drop=True)

            if len(set(start)) < n_p_total:
                raise ValueError(f'You must provide a value for every combination of {parameter_columns}.')

            if len(start) <= 1:
                continue
            for i in range(1, len(start)):
                e = end[i-1]
                s = start[i]
                if e > s or s == start[i-1]:
                    raise ValueError(f'Parameter data must not contain overlaps. Parameter {p} '
                                     f'contains overlapping data.')
                if e < s:
                    raise NotImplementedError(f'Interpolation only supported for parameter columns '
                                              f'with continuous bins. Parameter {p} contains '
                                              f'non-continuous bins.')


class Order0Interp:
    """A callable that returns the result of order 0 interpolation over input data.

    Attributes
    ----------
    data :
        The data from which to build the interpolation. Contains
        categorical_parameters and continuous_parameters.
    parameter_columns :
        Column names to be used as parameters in Interpolation.
    """
    def __init__(self, data, parameter_columns: ParameterType, value_columns: List[str], extrapolate: bool):
        """

        Parameters
        ----------
        data :
            Data frame used to build interpolation.
        parameter_columns :
            Parameter columns. Should be of form (column name used in call,
            column name for left bin edge, column name for right bin edge)
            or column name. If given as single column name, assumed to be
            midpoint of bin and continuous bins created. Assumes left bin
            edges are inclusive and right exclusive.
        extrapolate :
            Whether or not to extrapolate beyond the edge of supplied bins.

        """
        check_data_complete(data, parameter_columns)
        self.data = data.copy()
        self.value_columns = value_columns
        self.extrapolate = extrapolate

        # (column name used in call, col name for left edge, col name for right):
        #               [ordered left edges of bins], max right edge (used when extrapolation not allowed)
        self.parameter_bins = {}

        for p in parameter_columns:
            left_edge = self.data[p[1]].drop_duplicates().sort_values()
            max_right = self.data[p[2]].drop_duplicates().max()

            self.parameter_bins[tuple(p)] = {'bins': left_edge.tolist(), 'max': max_right}

    def __call__(self, interpolants: pd.DataFrame) -> pd.DataFrame:
        """Find the bins for each parameter for each interpolant in interpolants
        and return the values from data there.

        Parameters
        ----------
        interpolants:
            Data frame containing the parameters to interpolate..

        Returns
        -------
        pd.DataFrame
            A table with the interpolated values for the given interpolants.

        """
        # build a dataframe where we have the start of each parameter bin for each interpolant
        interpolant_bins = pd.DataFrame(index=interpolants.index)

        merge_cols = []
        for cols, d in self.parameter_bins.items():
            bins = d['bins']
            max_right = d['max']
            merge_cols.append(cols[1])
            interpolant_col = interpolants[cols[0]]
            if not self.extrapolate and (interpolant_col.min() < bins[0] or interpolant_col.max() >= max_right):
                raise ValueError(f'Extrapolation outside of bins used to set up interpolation is only allowed '
                                 f'when explicitly set in creation of Interpolation. Extrapolation is currently '
                                 f'off for this interpolation, and parameter {cols[0]} includes data outside of '
                                 f'original bins.')
            bin_indices = np.digitize(interpolant_col, bins)
            # digitize uses 0 to indicate < min and len(bins) for > max so adjust to actual indices into bin_indices
            bin_indices = [x-1 if x > 0 else x for x in bin_indices]

            interpolant_bins[cols[1]] = [bins[i] for i in bin_indices]

        index = interpolant_bins.index

        interp_vals = interpolant_bins.merge(self.data, how='left', on=merge_cols).set_index(index)
        return interp_vals[self.value_columns]





