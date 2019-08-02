.. _lookup_concept:

=============
Lookup Tables
=============

Simulations tend to require a large quantity of data to run. ``vivarium``
provides the :class:`Lookup Table` abstraction to ensure that accurate data can
be retrieved when it's needed.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The Lookup Table
----------------

A :class:`Lookup Table` for a quantity is a callable object that is built from a
scalar value or a :class:`pandas.DataFrame` of data points that describes how
the quantity varies with other variable(s). It is called with a
:class:`pandas.Index` as a function parameter which represents a simulated
population as discussed in the :ref:`population concept <population_concept>`.
When called, the :class:`Lookup Table` simply returns appropriate values of the quantity
for the population it was passed, interpolating if necessary or extrapolating
if configured to. This behavior represents the standard interface for asking for
data about a population in a simulation.

Construction Parameters
~~~~~~~~~~~~~~~~~~~~~~~

A lookup table is defined by variables called keys, parameters, and values. These
variables define the groups within which data is specific, the data

A lookup table is instantiated with column names corresponding to variables
that define groups of the population which are used to query an internal
`:term: population view`
when the table itself is called.  These column names correspond to keys, parameters and values.
It's important to keep these sets of variables straight when discussing these tables.
For the following examples, refer to the table of data.

A key is a categorical variable within which an interpolation takes place. Commonly
this is sex data, as in the data above. Functionally speaking, this results in
a different interpolation being performed for each distinct sex value. A parameter is a variable that
defines the independent

Keys
    lorem ipsum

Parameters
    lorem ipsum

Values
    lorem ipsum


    key is categorical within which an interpolation takes place
    parameter is the variable that defines the bins
    interpolant is the value you stick in
    "value" is what you get back

A LookupTable maintains its own `population view` and collects queries the population
itself according to the columns it was created with.

This is difficult to understand in words and potentially easier using a scenario
with numbers. In the below table, blah blah blah.





Interpolation
-------------

If a lookup table was constructed with a scalar value or values, the lookup call
trivially returns the same scalar(s) back for any population passed in. However, if the
lookup table was instead created with a :class:`pandas.DataFrame` of varying data,
the lookup will perform interpolation, which is an important feature of lookup tables.
Interpolation is the process of estimating values for unspecified parameters within the bounds of
the parameters we have defined in the lookup table. Currently, the most common
case is the data is defined for bins, and non-binned data is passed in. Then, the
interpolation simply finds the correct bin a value belongs to. Please see the
:ref:`interpolation concept note <population_concept>` for more in-depth information
about the kinds of interpolation performed by the lookup
table.

Extrapolation
-------------

Previously, we discussed interpolation as the process of estimating data within
the bounds defined by our lookup table. What would happen if we wanted data outside
of this range? Estimating such data is called extrapolation, and it can be performed using a
lookup table as well. Extrapolation is a configurable option that, when enabled,
allows a lookup data to provide values outside of the range it was created with.
This is done by extending the edge points outwards to encompass outside points.
Configuring it in a model specification looks like this:


