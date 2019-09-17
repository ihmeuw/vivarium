.. _population_concept:

=====================
Population Management
=====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

``Vivarium`` is an agent-based simulation so we must keep track of agents, or
:term:`simulants`, somewhere. Our simulation needs the ability to create these
simulants, give them state and update it, and provide access to
them to components in the simulation that wish to perform interesting actions.
We call the set of simulants the population, and the tooling to support working
with them is the population management system.

controls access

- Pop system manages the pop of simulants -- represented by the state table
- It provides ways to modify and view the state table

The population system manages the population of simulants in a simulation and
supplies the interface for looking at it and modifying it.


The State Table
---------------

The core representation of the simulants and their state information in a
``Vivarium`` simulation is a pandas DataFrame known as the state table.

Under this representation, rows correspond to simulants while columns correspond
to attributes or categories of state, like age, sex or blood pressure. The
PopulationManager holds this state table and controls access to it.





The population of simulants and their state information in a ``Vivarium``
simulation is maintained in a pandas DataFrame known as the state table.
In this table, rows correspond to simulants and columns correspond to attributes
or categories of state, like sex, age or blood pressure. The PopulationManager
holds this table and controls access to it. It directly controls the addition
of rows (simulants) and columns (attributes) to the state table and allows
access to the values in the state table, enabling updates to values that already
exist.



- Can modify existing values or create new ones
    - Since it is a table, we can think of new values along two axes - columns
      and rows. Columns are new categories of state, rows are new simulants.
- We can modify it along these axes and filter our view along these axes,
  as well. How do we do that ?

    - registering simulant initializers and providing the initializer functoin
    - providing views that optional filter by rows or columns, and can declare
      creation of columns
    - columns in the state table represent an important resource, to be detailed
      in the forthcoming resource documentation.



<<TODO: image of state table w/ expanding rows and columns >>


Viewing and Updating Simulants
------------------------------

We will put off discussing how simulants are made and first talk about how
simulant state is viewed and updated. The mechanism of viewing and updating
state is the PopulationView.

It provides access to the state table through controlled "views"


Creating Simulants
-------------------

Simulants are are introduced to the simulation using a function that takes
the number of new simulants as it's parameter. This function, known as the
simulant creator, is provided by the population management system. The population system


There are two distinct execution states in which simulants can be created.
The population initialization state during the setup phase, and the main event
loop.


creating simulants is quite similar to updating state. However, we are allowed
to create columns on initialization.


