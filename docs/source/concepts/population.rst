.. _population_concept:

=====================
Population Management
=====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Since ``Vivarium`` is an agent-based simulation framework, managing a group of
:term:`simulants` and their attributes is a critical task. Fundamentally, to run
a simulation we need to be able to create new simulates, update their state
attributes, and facilitate access to their state so that :term:`components` in
the simulation can do interesting things based on it. The tooling to support
working with our simulant population is called the population management system.

The State Table
---------------

The core representation of simulants and their state information in ``Vivarium``
is a :class:`pandas.DataFrame <Data Frame>` known as the state table. Under this
representation rows represent simulants while columns correspond to state
attributes like age, sex or systolic blood pressure. These columns represent one
of several important resources within ``Vivarium`` that other components can
draw on. Each of the actions we need to be able to take correspond to a
manipulation of this state table. The addition of new simulants is the creation
of rows, the creation of new state attributes is the creation of columns, and
the reading and updating of state is reading and updating the dataframe itself.

<<TODO: image of state table w/ expanding rows and columns>>

Population Views
----------------

The population manager holds the state table directly and tightly controls read
and write access to it through a structure it provides known as a population
view. A population view itself represents a subset of columns and rows from the
state table. Through a view, components can read, update, or, under the right
circumstances, create new state in the state table.

Views are created on-demand for components in a simulation by specifying a set
of columns and an optional query string to the population manager interface. The
columns dictate the subset of the state table that is viewable and modifiable
while the query string is a filter on the simulants returned. The view itself is
callable and accepts an index, which is the simulants to be viewed. It also
provides an update method that accepts a dataframe and will replace values in
the state table according to column and index. Only the columns that the view
was created with can be updated in this way. The only exception is at simulant
initialization time, when initial state must be created.

Population views can themselves create subviews through the subview method. This
generates a new population view that is constrained by it's parents columns and
query string in addition to whatever arguments it is passed, with the
requirement that it's columns must be a subset of its parent view's.

<<TODO: Potentially a view to subview picture showing additional constraining>>

Creating Simulants
------------------

The population view pattern also underlies the creation of simulants, the only
difference being that when simulations are being initialized for the first time,
it is acceptable to create columns in the state table via update that don't
already exist.

The Simulant Creator Function
+++++++++++++++++++++++++++++

Simulants are are introduced to the simulation using a function that takes the
number of new simulants as its parameter. This function, known as the simulant
creator, is provided by the population manager interface and is used by the
simulation entrypoint to initialize the population. It can also be used by
components that want to introduce new simulants over the course of a simulation,
such as a fertility component that models births. This means there are two
distinct execution states in which simulants can be created: The population
initialization state during the setup phase, and the main event loop.

The simulant creator function first adds rows to the state table. It then loops
through a set of functions that have been registered to it as population
intitializers via `initializes_simulants`, passing in the index of the newly
created simulants. These functions generally proceed by using population views
to dictate the state of the newly created simulants they are responsible for.
It is the only time creating columns in the state table is acceptable.
