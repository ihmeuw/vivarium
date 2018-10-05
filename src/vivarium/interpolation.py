import warnings

import pandas as pd
from scipy import interpolate
from typing import Sequence

class Interpolation:
    def __init__(self, data: pd.DataFrame, categorical_parameters: Sequence[str],
                 continuous_parameters: Sequence[str], order: int):

        self.key_columns = categorical_parameters
        self.parameter_columns, self._data = validate_parameters(data, continuous_parameters, order)

        # FIXME: allow for more than 2 parameter interpolation
        if len(self.parameter_columns) not in [1, 2]:
            raise ValueError("Only interpolation over 1 or 2 variables is supported")

        # These are the columns which the interpolation function will approximate
        value_columns = sorted(self._data.columns.difference(set(self.key_columns)|set(self.parameter_columns)))
        if not value_columns:
            raise ValueError(f"No non-parameter data. Available columns: {self._data.columns}, "
                             f"Parameter columns: {set(self.key_columns)|set(self.parameter_columns)}")

        if self.key_columns:
            # Since there are key_columns we need to group the table by those
            # columns to get the sub-tables to fit
            sub_tables = self._data.groupby(list(self.key_columns))
        else:
            # There are no key columns so we will fit the whole table
            sub_tables = {None: self._data}.items()

        self.interpolations = {}

        for key, base_table in sub_tables:
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

    def __call__(self, population):

        self. _validate_call_data(population)

        if self.key_columns:
            sub_tables = population.groupby(list(self.key_columns))
        else:
            sub_tables = [(None, population)]
        result = pd.DataFrame(index=population.index)
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

        if len(result.columns) == 1:
            return result[result.columns[0]]

        return result

    def _validate_call_data(self, data):
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f'Interpolations can only be called on pandas.DataFrames. You'
                            f'passed {type(data)}.')

        if not set(self.parameter_columns) <= set(data.columns.values.tolist()):
            raise ValueError(f'The continuous parameter columns with which you built the Interpolation must all'
                             f'be present in the data you call it on. The Interpolation has key'
                             f'columns: {self.parameter_columns} and your data has columns: '
                             f'{data.columns.values.tolist()}')

        if self.key_columns and not set(self.key_columns) <= set(data.columns.values.tolist()):
            raise ValueError(f'The key (categorical) columns with which you built the Interpolation must all'
                             f'be present in the data you call it on. The Interpolation has key'
                             f'columns: {self.key_columns} and your data has columns: '
                             f'{data.columns.values.tolist()}')

    def __repr__(self):
        return "Interpolation()"





def validate_parameters(data, continuous_parameters, order):
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
    return out, data
