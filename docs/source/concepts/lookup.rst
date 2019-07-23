.. _lookup_concept:

===============================
Lookup Tables and Interpolation
===============================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

we need data. lookup table gives you data for a pop. this give us a nice,
easy interface.

lookup table is callable object that takes an index representing a pop. spits
back values

we may need to interpolate. when do we interpolate? when we made a table with
more than a scalar
    if scalar, just gives value
    if not, we have a problem. what data is accurate? interpolate. 0 order, don't support higher

we do zero order interpolation. it works like this
    with bins -- pic
    with values -- pic

we can also extrapolate -- get data beyond our bounds.  under zero -order this
doesn't change much. the nearest bin or point is taken as the value, it just
may be far away.


== header

Simulations tend to require a large quantity of data to run. ``vivarium``
provides the :class:`LookupTable` abstraction to ensure that accurate data can
be retrieved when it's needed.

== The lookuptable

A :class:`LookupTable` is a callable object that is built from a scalar
value or a :class:`pandas.DataFrame` of values and accepts a
:class:`pandas.Index`. As discussed in the :ref:`population concept section <population_concept>`
the pandas index is the fundamental representation of the simulated population.
When called, the LookupTable simply returns values for the population it was
passed -- that's it! In doing so, it may perform `interpolation` if the data is
non-specific. This behavior represents the standard interface for asking for data about a population
in a simulation.

There are several different sets of variables with different meanings that are
important to keep straight when discussing these tables.




== Interpolation

If a lookup table was constructed with a scalar value or values, the lookup call
trivially returns the same scalar(s) back. However, if the lookup table was
instead created with a set of keys, parameters and values, the lookup call is
more complicated. For instance, take the following series of age bins and life expectencies. If we tracked
age as a floating point number, then passing in a simulated population with
there ages to a lookup table and

This lookup requires interpolation, which is an important feature of lookup
tables. Interpolation is the process of estimating this data within the bounds of
the data we have defined in the lookup table. There are many ways to perform
this estimation. Currently, ``vivarium`` supports zero-order interpolation only.

== Zero - order interpolation

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

== extrapolation

Previously, we discussed interpolation as the process of estimating data within
the bounds defined by our lookup table. What would happen if we wanted data outside
of this range? This is called extrapolation, and it can be performed using a
lookup table as well. For the zero-order case this doesn't change much. The
lookup table returns the nearest edge value, though this value could be quite
far away.  TODO: is it parameter or value or what


== details
keeps track of column names and its own population view
can handle multiple value columns and can handle key columns that
are categorical variables interpolations should be specific to.

key is categorical within which an interpolation takes place
parameter is the variable that defines the bins
interpolant is the value you stick in
"value" is what you get back



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





