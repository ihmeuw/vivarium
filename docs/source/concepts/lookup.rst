.. _lookup_concept:

=============
Lookup Tables
=============

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
scalar value or a :class:`pandas.DataFrame` of data points that describes how the
quantity varies with other variable(s). It is called with a :class:`pandas.Index`
as a function parameter which represents a simulated population as discussed in the
:ref:`population concept note <population_concept>`. When called, the
LookupTable simply returns appropriate values of the quantity for the population
it was passed, interpolating if necessary or extrapolating if configured to. This behavior represents the standard
interface for asking for data about a population in a simulation.

Construction Parameters
~~~~~~~~~~~~~~~~~~~~~~~
A lookup table is instantiated with column names corresponding to the variables
that define groups of the population, that are used to query an internal population view
when the table itself is called.  These column names correspond to keys, parameters and values.
It's important to keep these sets of variables straight when discussing these tables.
For the following examples, refer to the table of data.

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

Please see the :ref:`interpolation concept note <population_concept>` for more
in-depth information about the kinds of interpolation performed by the lookup
table.

Extrapolation
-------------

Previously, we discussed interpolation as the process of estimating data within
the bounds defined by our lookup table. What would happen if we wanted data outside
of this range? This is called extrapolation, and it can be performed using a
lookup table as well. For the zero-order case this doesn't change much. The
lookup table returns the nearest edge value, though this value could be quite
far away.  TODO: is it parameter or value or what

extrapolation is configurable. in a model specification this looks like this.
