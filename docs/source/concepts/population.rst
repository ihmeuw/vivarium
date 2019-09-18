.. _population_concept:

=====================
Population Management
=====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Since ``Vivarium`` is an agent-based simulation framework, managing a group of :term:`simulants` is a critical task. Our simulations need the ability to create simulants, update their state, and furnish access to the simulants to components in the simulation so they can do interesting things.  We call the set of simulants the population, and the tooling to support working with them is the population management system.

The State Table
---------------

The core representation of simulants and their state information in ``Vivarium`` is a pandas DataFrame known as the state table. In this representation, rows correspond to simulants while columns correspond to state: attributes like age, sex or systolic blood pressure. These columns represent one of several important resources within ``Vivarium`` that other components can draw on to do things..

The population manager holds the state table directly and tightly controls access to it. Since the state table is two-dimensional, we can think about altering it in two ways, by adding rows ( new simulants) or by adding columns (new categories of state).

In general, we are interested in 

It directly controls the addition of rows (simulants) and columns (attributes) to the state table and allows access to the values in the state table, enabling updates to values that already exist.



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
simulant state is viewed and updated because the underlying mechanism is the same. The population manager interface provides a function for generating PopulationView objects that can be used to retrieve and update state information for simulants. A population view is rigidly controlled. It is instantiated with a set of columns that dictate what data is retrieved, and only that data can be updated. The only exception to this is when simulants are being created.

Population views optionall

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


