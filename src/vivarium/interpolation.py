import warnings

import pandas as pd
from scipy import interpolate
from typing import Union, List, Tuple
from itertools import product


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
            Column names to be used as continuous parameters in Interpolation.
        order :
            Order of interpolation.
        """

    def __init__(self, data: pd.DataFrame, categorical_parameters: Union[List[str], Tuple[str]],
                 continuous_parameters: Union[List[str], Tuple[str]], order: int):

        self.key_columns = categorical_parameters
        self.parameter_columns, self._data, value_columns = validate_parameters(data, categorical_parameters,
                                                                                continuous_parameters, order)

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
            # For each permutation of the key columns build interpolations
            self.interpolations[key] = {}
            for value_column in value_columns:
                # For each value in the table build an interpolation function
                if len(self.parameter_columns) == 2:
                    # 2 variable interpolation
                    if order == 0:
                        x = base_table[list(self.parameter_columns)]
                        y = base_table[value_column]
                        func = interpolate.NearestNDInterpolator(x=x.values, y=y.values)
                    else:
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
                    if order == 0:
                        func = interpolate.interp1d(x, y, kind='zero', fill_value='extrapolate')
                    else:
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
        result = pd.DataFrame(index=interpolants.index)
        for key, sub_table in sub_tables:
            if sub_table.empty:
                continue
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

    # FIXME: allow for more than 2 parameter interpolation
    if len(continuous_parameters) not in [1, 2]:
        raise ValueError("Only interpolation over 1 or 2 variables is supported")

    out = []
    for p in continuous_parameters:
        if len(data[p].unique()) > order:
            out.append(p)
        else:
            warnings.warn(f"You requested an order {order} interpolation over the parameter {p}, "
                          f"however there are only {len(data[p].unique())} unique values for {p}"
                          f"which is insufficient to support the requested interpolation order."
                          f"The parameter will be dropped from the interpolation.")
            data = data.drop(p, axis='columns')

    # These are the columns which the interpolation function will approximate
    value_columns = sorted(data.columns.difference(set(categorical_parameters) | set(continuous_parameters)))
    if not value_columns:
        raise ValueError(f"No non-parameter data. Available columns: {data.columns}, "
                         f"Parameter columns: {set(categorical_parameters)|set(continuous_parameters)}")
    return out, data, value_columns


def validate_call_data(data, key_columns, parameter_columns):
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f'Interpolations can only be called on pandas.DataFrames. You'
                        f'passed {type(data)}.')

    if not set(parameter_columns) <= set(data.columns.values.tolist()):
        raise ValueError(f'The continuous parameter columns with which you built the Interpolation must all'
                         f'be present in the data you call it on. The Interpolation has key'
                         f'columns: {parameter_columns} and your data has columns: '
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

    param_edges = [p for p in parameter_columns if isinstance(p, (Tuple, List))]
    param_points = [p for p in parameter_columns if isinstance(p, str)]

    # FIXME: There has to be a cleaner, faster, less horrible way to do this
    # check no overlaps/gaps
    for p in param_edges:
        other_params = param_points + [p_ed[0] for p_ed in param_edges if p_ed != p]
        if other_params:
            sub_tables = data.groupby(list(other_params))
        else:
            sub_tables = {None: data}.items()

        n_p_total = len(set(data[p[0]]))

        for _, table in sub_tables:

            param_data = table[[p[0], p[1]]].copy().sort_values(by=p[0])
            start, end = param_data[p[0]].reset_index(drop=True), param_data[p[1]].reset_index(drop=True)

            if len(set(start)) < n_p_total:
                raise ValueError(f'You must provide a value for every combination of {parameter_columns}')

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

    # check all combos - because we know there are no overlaps/repeats, we can just check the numbers match
    #params = param_points + [p[0] for p in param_edges]
    #combos = list(product(*[set(data[p]) for p in params]))

    #if len(combos) > data.shape[0]:
    #    raise ValueError(f'You must provide a value for every combination of {parameter_columns}')



