.. _lookup_concept:

=============
Lookup Tables
=============

Simulations tend to require a large quantity of data to run. ``vivarium``
provides the :class:`Lookup Table <vivarium.framework.lookup.LookupTable>`
abstraction to ensure that accurate data can be retrieved when it's needed.

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
When called, the :class:`Lookup Table <vivarium.framework.lookup.LookupTable>`
simply returns appropriate values of the quantity for the population it was
passed, interpolating if necessary or extrapolating if configured to. This
behavior represents the standard interface for asking for data about a
population in a simulation.

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

Along with data about these variables, A lookup table is instantiated with the
corresponding column names which are used to query an internal population view
when the table itself is called. This means the lookup table only needs to be
called with a population index. It gathers the population information it needs
itself. It also means the data must be available in the
:term:`population state table <State Table>` with the same column name.

This is difficult to describe textually and may be easier understood with
numbers. In the table below is an example of (unrealistic) data that could be
used to create a lookup table for a quantity of interest about a population,
in this case, Body Mass Index (BMI). When called, the lookup table will return
values of BMI for the simulants defined by the population index. See the example
below.

======  =========  =======  ======
Key         Parameter       Value
------  ------------------  ------
sex     age_start  age_end   BMI
======  =========  =======  ======
Male    0          20       20
Male    20         40       25
Male    40         60       30
Male    60         100      27
Female  0          20       20
Female  20         40       25
Female  40         60       30
Female  60         100      27
======  =========  =======  ======

Example Usage
~~~~~~~~~~~~~

The following is an example of creating and calling a lookup table in an
`interactive setting <_interactive_tutorial>`. The interface and process are the
same when integrating a lookup table into a :term:`component <Component>`, which
is primarily how they are used. Assuming you have a valid simulation object
named `sim` and the data from the above table in a pandas dataframe named
`data`, you can construct a lookup table in the following way, using the
interface from the builder.

.. code-block:: python

      # value_columns implicitly set to remaining columns
    > bmi = sim.builder.lookup.build_table(data, key_columns=['sex'], parameter_columns=[('age', 'age_start', 'age_end')])
    > population = sim.get_population()
    > bmi(population.index).head()  # returns BMI values for the population

      0     20.0
      1     20.0
      2     30.0
      3     27.0
      4     25.0
      Name: BMI, dtype: float64

.. note::
    constructing a lookup table currently requires your data meet specific
    conditions. These are a consequence of the method the lookup table uses to
    arrive at the correct data. Specifically, your parameter columns must
    represent bins and they must overlap.

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

Specifying Options in the Model Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configuring interpolation and extrapolation in a model specification is
straightforward. Currently, the only acceptable value for order is `0`.
Extrapolation can be turned on and off.

.. code-block:: yaml

    configuration:
        interpolation:
            order: 0
            extrapolate: True
