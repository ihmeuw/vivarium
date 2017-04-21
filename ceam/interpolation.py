import pandas as pd
import numpy as np

from scipy import interpolate

class Interpolation:
    def __init__(self, data, categorical_parameters, continuous_parameters, func=None, order=1):
        self.key_columns = categorical_parameters
        self.parameter_columns = continuous_parameters
        self.func = func

        if len(self.parameter_columns) not in [1, 2]:
            raise ValueError("Only interpolation over 1 or 2 variables is supported")
        if len(self.parameter_columns) == 1 and order == 0:
            raise ValueError("Order 0 only supported for 2d interpolation")

        # These are the columns which the interpolation function will approximate
        value_columns = sorted(data.columns.difference(set(self.key_columns)|set(self.parameter_columns)))

        if self.key_columns:
            # Since there are key_columns we need to group the table by those
            # columns to get the sub-tables to fit
            sub_tables = data.groupby(self.key_columns)
        else:
            # There are no key columns so we will fit the whole table
            sub_tables = {None: data}.items()

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
                    x = base_table[self.parameter_columns[0]]
                    y = base_table[value_column]
                    func = interpolate.InterpolatedUnivariateSpline(x, y, k=order)
                self.interpolations[key][value_column] = func

    def __call__(self, *args, **kwargs):
        # TODO: Should be more defensive about this
        if len(args) == 1:
            # We have a dataframe
            df = args[0]
        else:
            # We have parameters for a single invocation
            df = pd.DataFrame(kwargs)

        if self.key_columns:
            sub_tables = df.groupby(self.key_columns)
        else:
            sub_tables = [(None, df)]

        result = pd.DataFrame(index=df.index)
        for key, sub_table in sub_tables:
            funcs = self.interpolations[key]
            parameters = tuple(sub_table[k] for k in self.parameter_columns)
            for value_column, func in funcs.items():
                out = func(*parameters)
                # This reshape is necessary because RectBivariateSpline and InterpolatedUnivariateSpline return results
                # in slightly different shapes and we need them to be consistent
                if out.shape:
                    result.loc[sub_table.index, value_column] = out.reshape((out.shape[0],))
                else:
                    result.loc[sub_table.index, ouput_column] = out

        if self.func:
            return self.func(result)
        else:
            if len(result.columns) == 1:
                return result[result.columns[0]]
            else:
                return result
