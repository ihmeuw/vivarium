.. _lifecycle_concept:

========================
The Simulation Lifecycle
========================

A ``vivarium`` simulation has many important lifecycle phases.

It begins with a user interacting with one of the simulation
:ref:`entry points <entry_points_concept>`. The simulation then goes through an
initialization cycle that creates all the :term:`components <Component>`,
and compiles :term:`configuration information <Configuration>`.  After
initialization, the simulation enters the setup phase, where each component
in turn has a special ``setup`` method called with the simulation
:ref:`builder <builder_concept>`.  During the setup, components ask for
services from the framework and register themselves with various simulation
subsystems.  There is then a post setup phase where the simulation framework
does some cleanup.

The simulation then enters its main processing section. The population
is initialized, then the time step loop begins. Once the loop completes,
results are calculated, gathered, and written out.

.. contents::
   :depth: 2
   :local:
   :backlinks: none

Initialization
--------------

The initialization phase of ``vivarium`` lasts from when a user first interacts
with a simulation :ref:`entry point <entry_points_concept>` to when the
``__init__`` method of the :class:`vivarium.framework.engine.SimulationContext`
completes.  In it, the following things happen (roughly in this order):

1. Simulation outputs are configured. This is usually just setting up some
   directories for the final results and the simulation logs.
2. :term:`Model specification <Model Specification>` defaults are collected and
   compiled along with the model specification file (if provided) into a
   :class:`vivarium.config_tree.ConfigTree` object. Importantly,
   default :term:`configuration <Configuration>` information from specific
   components is not compiled into the :class:`vivarium.config_tree.ConfigTree`
   at this point.
3. A :class:`vivarium.framework.plugins.PluginManager` is generated around the
   :term:`plugins <Plugin>` section of the model specification.  The plugin
   manager is responsible for parsing the plugin section and instantiating
   plugin controllers and interfaces for the framework.
4. If :term:`components <Component>` are not provided directly by the user in
   an interactive setting, the
   :class:`vivarium.framework.components.ComponentConfigurationParser` is
   created to parse and instantiate them from the
   :term:`Model specification <Model Specification>`.
5. The ``__init__`` method of the
   :class:`vivarium.framework.engine.SimulationContext` is called with
   the current :term:`configuration <Configuration>`, the instantiated
   components, and the plugin manager.  This creates the
   :ref:`builder <builder_concept>` which provides an interface to all the
   simulation subsystems during the next phase of the simulation lifecycle.
   It also registers all the simulation components with the
   :class:`vivarium.framework.components.ComponentManager`.

At this point, all input arguments have been parsed and all top-level
components have been instantiated.  This is a useful phase in the simulation
lifecycle because you can typically modify what components are in the system
or how they are configured without any consequences.

Setup
-----

In this stage, the framework moves to setting up the
:term:`components <Component>`. For each top-level component, the framework
applies any :term:`configuration <Configuration>` defaults of the component.
Next, it calls a special ``setup`` on each component, providing each component
access to the simulation :ref:`builder <builder_concept>` which allows the
components to request services like :ref:`randomness <crn_concept>` or views
into the :term:`population state table <State Table>` or to register themselves
with various simulation subsystems. Setting up components may also involve
loading data, registering or getting :ref:`pipelines <values_concept>`,
creating :ref:`lookup tables <lookup_concept>`, and registering
:ref:`population initializers <population_concept>`, among other things.
The specifics of this are determined by the ``setup`` method on each component
- the framework itself simply calls that method with a
:class:`vivarium.framework.engine.Builder` object.  Part of component setup
may sometimes spawn sub-components, so this process continues until all
components are setup.

Post-setup
----------

This is a small phase that exists in the simulation mainly so that framework
:term:`managers <Plugin>` can coordinate shared state and do any necessary
cleanup.  This is the first actual :ref:`event <event_concept>` emitted by
the simulation framework.  Normal ``vivarium`` :term:`components <Component>`
should never listen for this event.  This may be enforced at a later date.

Population Initialization
-------------------------

It's not until this stage that the framework actually generates the base
:ref:`population <population_concept>` for the simulation. Here, the framework
rewinds the simulation :ref:`clock <time_concept>` one time step and generates
the population.  This time step fence-posting ensures that
:term:`simulants <Simulant>` enter the simulation on the correct start date.
Note that this rewinding of the clock is purely what it sounds like - there is
no concept of a time step being taken here. Instead, the clock is literally
reset back the duration of one time step. Once the simulant population is
generated, the clock is reset to the simulation start time, again by changing
the clock time only without any time step being taken.

The Main Event Loop
-------------------

At this stage, all the preparation work has been completed and the framework
begins to move through the simulation. This occurs as an
:ref:`event loop <event_concept>`. The framework emits a series of events for
each :ref:`time step <time_concept>`:

1. *time_step__prepare*
   A phase in which simulation :term:`components <Component>` can do any
   work necessary to prepare for the time step.
2. *time_step*
   The phase in which the bulk of the simulation work is done.  Simulation
   state is updated.
3. *time_step__cleanup*
   A phase for simulation components to do any post time step cleanup.
4. *collect_metrics*
   A life-cycle phase specifically reserved for computing and recording
   simulation outputs.

By listening for these events, individual components can perform actions,
including manipulating simulants. This sequence of events is repeated until
the simulation clock passes the simulation end time.

.. note::

    We have multiple sources of time during this process. The
    :class:`vivarium.framework.engine.SimulationContext` itself holds onto a
    clock. This simulation clock is the actual time in the simulation. Events
    (including e.g., *time_step*) come with a time as well. This time is the
    time at the start of the next time step, that is, the time when any changes
    made during the loop will happen.


Finalization
------------

The final stage in the simulation life cycle is fittingly enough, finalization.
At this stage, the *simulation_end* :ref:`event <event_concept>` is emitted to
signal that the event loop has finished and the
:ref:`state table <population_concept>` is final. At this point, final
simulation outputs are safe to compute.
