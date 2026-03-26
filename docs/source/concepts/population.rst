.. _population_concept:

=====================
Population Management
=====================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Since ``Vivarium`` is an agent-based simulation framework, managing a group of
:term:`simulants <Simulant>` and their :term:`attributes <Attribute>` is a critical task.
Fundamentally, to run a simulation we need to be able to create new simulates,
update their state attributes, and facilitate access to their state so that
:term:`components <Component>` in the simulation can do interesting things
based on it. The tooling to support working with our simulant population is
called the population management system.

The Population State Table
--------------------------

The core representation of simulants and their state information in ``Vivarium``
is a dynamically-generated :class:`pandas.DataFrame` known as the population state
table (or just "state table"). Under this representation, rows represent simulants
while columns correspond to attributes like age, sex or systolic blood pressure.
These columns represent one of several important resources within ``Vivarium`` that
other components can draw on. Each of the actions we need to be able to take correspond
to a manipulation of this state table. The addition of new simulants is the creation
of rows, the creation of new attributes is the creation of columns, and the reading
and updating of state is reading and updating the dataframe itself.

<<TODO: image of state table w/ expanding rows and columns>>

Attributes
----------

Attributes are the fundamental characteristics of a population and are represented
by columns in the population state table. They are a particular type of :term:`values <Value>`
that are produced by on-demand by :term:`attribute pipelines <Attribute Pipeline>`.
When a component requires the state table (or some subset of it), each attribute
requested is calculated via its corresponding attribute pipeline and returned in
tabular form. For example, when a component requests the entire population's age,
the "age" attribute pipeline calculates the age of all simulants and returns a
``pandas.Series`` of age values.

.. note::
   The population system is distinct from the :ref:`values system documentation <values_concept>`
   although they are intimately related. While the values system is responsible for
   populating the columns of the state table with attributes, the population system
   is responsible for managing and providing access to said state table.

Population Views
----------------

As mentioned above, columns in the state table are dynamically generated via attribute
pipelines as needed. The population manager holds this logic and tightly controls
read and write access to it through a structure it provides known as a "population
view". A population view itself provides access to a subset of columns and rows
from the state table as well as any :term:`private columns <Private Column>` created
by the component the view is attached to. Through a view, components can read, update,
or, under the right circumstances, create new state in the state table.

Views are created for components in a simulation by specifying the component
needing it and an optional query string to the population manager interface. All
attributes are then viewable and the query string filters the simulants returned.
And as noted above, they also have read and write access to all private columns
created by the component they are attached to. This is how one might update the
source data for attributes, e.g. updating all simulants' ages on every time step.

There are several methods on a population view that facilitate working with the
state table, including ones to get the population index and attributes. There are
also two methods for writing to private columns:

- :meth:`~vivarium.framework.population.population_view.PopulationView.initialize`
  is used during simulant creation (both initial population and new cohorts) to
  write initial values for private columns.
- :meth:`~vivarium.framework.population.population_view.PopulationView.update`
  is used during the simulation to modify existing private column data. It takes
  the column name(s) and a modifier function that receives the current values and
  returns the updated values.

Filtering Simulants
+++++++++++++++++++

There are two types of filtering that can be applied when using a population view
to get attributes or private columns.

First, a ``query`` argument can be passed in to any of the population view's
:meth:`~vivarium.framework.population.population_view.PopulationView.get`,
:meth:`~vivarium.framework.population.population_view.PopulationView.get_frame`, or
:meth:`~vivarium.framework.population.population_view.PopulationView.get_filtered_index`
to filter the simulants returned for that specific call.

Second, if any components have registered an untracking query, untracked simulants
will be automatically filtered out. There is an optional ``include_untracked`` argument 
that defaults to False that can be used to bypass the untracked filtering if desired.

.. note::

    **Combining Queries**
    All types of queries are combined using the logical AND operator. Be sure to
    set up your query strings accordingly.

Untracking Simulants
++++++++++++++++++++

As mentioned above, there is a ``Vivarium`` concept of untracking simulants. Untracking
a simulant allows for automatic filtering of those simulants from population views
so that components can ignore them. This is useful to reduce computational overhead
when simulants are no longer relevant to the simulation, e.g. deceased individuals
or those who have aged beyond the scope of interest. A component can register a
tracked query via :meth:`vivarium.framework.population.interface.PopulationInterface.register_tracked_query`.

.. note::

    **Tracked Queries and Including Untracked Simulants**
    When a component wants to register a query to be used for filtering out untracked
    simulants, it registers the *tracked* query, i.e. the query that defines which
    simulants should be *kept*. This can perhaps be a bit confusing since we then
    decide to include or exclude *untracked* simulants when using population views.
    Despite this potential source of confusion, we feel it's more intuitive to think
    about the query in terms of who to keep and then the population view call in
    terms of who to exclude.

    For example, if a component wants to untrack simulants whohave died, it would
    register ``is_alive == True`` as a tracked query which tells ``Vivarium`` to
    **keep** simulants who are alive (and, conversely, filter out those who are not).
    Then, when using a population view to data, we can decide whether or not to
    include untracked simulants or not (i.e. deceased ones).

Private Columns
---------------

We have mentioned private columns a few times now, but what exactly are they and
how do they differ from attributes (which can be thought of as "public" columns 
in the state table)? To start, keep in mind that attributes are produced by attribute
pipelines and *all* pipelines - attribute or otherwise - require a source of data
to operate on. One of the things that an attribute pipeline's source can be is
a column of data. All such attribute pipeline source columns are maintained in a
:class:`pandas.DataFrame` attached to the population manager, *but are only accessible
by population views attached to the component that created the source data in the
first place*. These columns are thus referred to as private columns.

Creating and Updating Private Columns
+++++++++++++++++++++++++++++++++++++

To create a private column to be used as a source for an attribute pipeline, a component
must register initializer methods during its setup. Any columns that are created
and passed to the population view's
:meth:`~vivarium.framework.population.population_view.PopulationView.initialize`
method within these methods will be automatically registered as private columns
for that component. The corresponding attribute pipelines will be registered
automatically as well.

To update private column data over the course of a simulation, a component can use
the population view's
:meth:`~vivarium.framework.population.population_view.PopulationView.update`
method, passing the column name(s) and a modifier function.

.. note::

    **Private columns vs attributes**
    The distinction between private columns and attributes can be confusing. It's
    important to remember that attributes are dynamically calculated as needed 
    (via attribute pipelines) and are readable by all components (via population
    views). Private columns, on the other hand, are static data stored in the population
    manager that are only readable and writable by the component that created them
    and serve as the source for their corresponding attributes.

    Private column data can be updated as needed by the owning component. These
    updates are then reflected in the attributes calculated from them the next time
    they are requested. For example, a component that creates an "age" private column
    (and thus and "age" attribute) instantiates the starting ages for all simulants
    at the start of the simulation. At each time step, the component can then update
    the private column by incrementing all ages by the duration of the time step.
    The next time any component then requests the "age" attribute, the updated ages
    will be returned since the source data was update.

Creating Attributes
-------------------

There are two ways to create attributes. The first, as described above, is to have
a component register an initializer method during its setup phase which creates
a private column. This private column will act as the source of data for its corresponding
attribute pipeline which is automatically registered as well. For example, if a 
component creates an "age" private column, and "age" attribute pipeline will be
automatically registered and so the "age" attribute will be available for use by
all components.

Not all attributes use a private column as their source, however. A component can
also register an attribute pipeline explicitly during its setup phase by calling
the values manager interface's :meth:`~vivarium.framework.values.interface.ValuesInterface.register_attribute_producer` 
or :meth:`~vivarium.framework.values.interface.ValuesInterface.register_rate_producer` methods.

Creating Simulants
------------------

The population view pattern also underlies the creation of simulants, the only
difference being that when simulations are being initialized for the first time,
it is acceptable to create columns in the state table via
:meth:`~vivarium.framework.population.population_view.PopulationView.initialize`
that don't already exist.

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
initializers via :meth:`~vivarium.framework.population.interface.PopulationInterface.register_initializer`,
passing in the index of the newly created simulants. These functions generally proceed
by using population views to dictate the state of the newly created simulants they
are responsible for. It is the only time creating columns in the state table is
acceptable.
