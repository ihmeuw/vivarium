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

A :class:`Lookup Table` for a quantity is a callable object that is built from
a scalar value or a :class:`pandas.DataFrame` of data points that describes how
the quantity varies with other variable(s). It is called with a
:class:`pandas.Index` as a function parameter which represents a simulated
population as discussed in the :ref:`population concept <population_concept>`.
When called, the :class:`Lookup Table` simply returns appropriate values of the
quantity for the population it was passed, interpolating if necessary or
extrapolating if configured to. This behavior represents the standard interface
for asking for data about a population in a simulation.

Construction Parameters
~~~~~~~~~~~~~~~~~~~~~~~

A lookup table is defined for a set of categorical variables, continuous
variables, and the values that depend on those variables. The lookup table
calls these variables keys, parameters, and values, respectively.

key
    A categorical variable, such as sex, that a quantity depends on.
parameter
    A continuous variable, such as age, that a quantity depends on. This data
    frequently represents bins for which values are defined.
value
    Known values of the quantity of interest, which vary with the keys and
    parameters.

A lookup table is instantiated with column names corresponding to these
variables which are used to query an internal `:term: population view` when the
table itself is called. This means the lookup table only needs to be called
with a population index. It gathers the population information it needs itself.

This is difficult to describe textually and may be easier understood with
numbers. In the table below is an example of data that could be used to create
a lookup table for a quantity of interest about a population, in this case,
XXX. When called, the lookup table will return values of XXX for the simulants
defined by the population index. See the example below.

.. :TODO: insert table

Example
~~~~~~~

.. :TODO: code snippet of instantiating and getting data from an interpolation table

Estimating Unknown Values
-------------------------

Interpolation
~~~~~~~~~~~~~

If a lookup table was constructed with a scalar value or values, the lookup
call trivially returns the same scalar(s) back for any population passed in.
However, if the lookup table was instead created with a
:class:`pandas.DataFrame` of varying data the lookup will perform interpolation
which is an important feature. Interpolation is the process of estimating
values for unspecified parameters within the bounds of the parameters we have
defined in the lookup table. Currently, the most common case arises when the
values are binned by the parameters. Then, the interpolation simply finds the
correct bin a value belongs to. Please see the
:ref:`interpolation concept note <population_concept>` for more in-depth
information about the kinds of interpolation performed by the lookup table.

Extrapolation
~~~~~~~~~~~~~

Previously, we discussed interpolation as the process of estimating data within
the bounds defined by our lookup table. What would happen if we wanted data 
outside of this range? Estimating such data is called extrapolation, and it can
be performed using a lookup table as well. Extrapolation is a configurable
option that, when enabled, allows a lookup data to provide values outside of
the range it was created with. This is done by extending the edge points
outwards to encompass outside points.

In the Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring interpolation and extrapolation in a model specification is
straightforward. Currently, the only acceptable value for order is `0`.
Extrapolation can be turned on and off.

.. code-block:: yaml
    configuration:
        interpolation:
            order: 0
            extrapolate: True

