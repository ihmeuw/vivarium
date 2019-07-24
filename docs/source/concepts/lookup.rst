.. _lookup_concept:

===============================
Lookup Tables and Interpolation
===============================

Simulations tend to require a large quantity of data to run. ``vivarium``
provides the :class:`LookupTable` abstraction to ensure that accurate data can
be retrieved when it's needed.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The Lookup Table
----------------

A :class:`LookupTable` for a quantity is a callable object that is built from a
scalar value or a :class:`pandas.DataFrame` of data points describing how the
quantity varies with other variable(s). It is called with a :class:`pandas.Index`
as a function parameter, representing a simulated population as discussed in the
:ref:`population concept note <population_concept>`. When called, the
LookupTable simply returns appropriate values of the quantity for the population
it was passed, interpolating if necessary. This behavior represents the standard
interface for asking for data about a population in a simulation.

Construction Parameters
~~~~~~~~~~~~~~~~~~~~~~~
A lookup table is instantiated with column names corresponding to the variables
that define groups of the population, that are used to query an internal population view
when the table itself is called.  These column names correspond to keys, parameters and values.
It's important to keep these sets of variables straight when discussing these tables.
For the following examples, refer to the table of data.

.. bokeh-plot::
    :source-position: none

    import random
    from datetime import date
    from random import randint

    from vivarium_inputs import get_age_bins

    import pandas as pd
    import numpy as np
    from bokeh.io import output_notebook
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import ColumnDataSource
    from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

    # example table
    output_file("data_table.html")

    # make arbitrary binned age data
    bins = [(x, x+5) for x in range(0, 90, 5)]
    values = [x[0] + random.random() * 10 // 1 for x in bins]

    data = dict(
            sex=([1] * len(bins)) + ([2] * len(bins)),
            value=2 * values
        )
    source = ColumnDataSource(data)

    columns = [
            TableColumn(field="sex", title="Sex (key)"),
            TableColumn(field="age", title="Age (parameter)"),
            TableColumn(field="value", title="Value (value)")
        ]
    data_table = DataTable(source=source, columns=columns, width=400, height=280)

    show(data_table)

A key is a categorical variable within which an interpolation takes place. Commonly
this is sex data, as in the data above. Functionally speaking, this results in
a different interpolation being performed for each distinct sex value. A parameter is a variable that
defines the independent

a value is a


    key is categorical within which an interpolation takes place
    parameter is the variable that defines the bins
    interpolant is the value you stick in
    "value" is what you get back

A LookupTable maintains its own `population view` and collects queries the population
itself according to the columns it was created with.

Interpolation
-------------

If a lookup table was constructed with a scalar value or values, the lookup call
trivially returns the same scalar(s) back for any population. However, if the
lookup table was instead created with a set of keys, parameters and values (as in the
example data above), the lookup action is more complicated.
~For instance, take the following series of age bins and life expectancies.~
If we tracked age as a floating point number, then passing in a simulated population with
there ages to a lookup table and

This lookup requires interpolation, which is an important feature of lookup
tables. Interpolation is the process of estimating this data within the bounds of
the data we have defined in the lookup table, and there are many ways to do
this in general. Currently, ``vivarium`` supports zero-order interpolation only.

Zero-order Interpolation
~~~~~~~~~~~~~~~~~~~~~~~~

Zero order interpolation is the only interpolation method supported by lookup
tables. It amounts to a nearest neighbor search. We will illustrate its behavior
with examples.

0 order interp accepts bins or points. 0 order on a point will result in nearest neighbor behavior,
or the point acting as the midpoint of a bin.

data may be defined over a range, so the task becomes finding the bin that
a simulant belongs to. Or, data may be defined at discrete points, so the task
becomes finding the nearest concrete data point.

This is actually not quite an estimation in the case of bins, it is a practical
matter of matching an observation to its bin.

Extrapolation
-------------

Previously, we discussed interpolation as the process of estimating data within
the bounds defined by our lookup table. What would happen if we wanted data outside
of this range? This is called extrapolation, and it can be performed using a
lookup table as well. For the zero-order case this doesn't change much. The
lookup table returns the nearest edge value, though this value could be quite
far away.  TODO: is it parameter or value or what

.. bokeh-plot::
    :source-position: none

    # example plot
    import random

    from bokeh.io import output_notebook
    from bokeh.plotting import figure, output_file, show

    output_file("test.html")
    p = figure(plot_width=800, plot_height=600)

    # make arbitrary binned age data
    bins = [(x, x+5) for x in range(0, 90, 5)]
    values = [x[0] + random.random() * 10 // 1 for x in bins]

    # add circles
    for bin, val in zip(bins, values):
        p.circle(bin[0], val, size=10, legend='left bin edge')
        p.circle(bin[1], val, fill_color='white', size=10, legend='right bin edge')

    # compose line path
    line_path = [pt for bin in bins for pt in bin]
    line_values = [val for duped in zip(values, values) for val in duped]

    # add interpolation lines
    p.line(line_path, line_values, line_width=2, legend='Data')

    p.legend[0].location = 'top_left'

    show(p)
