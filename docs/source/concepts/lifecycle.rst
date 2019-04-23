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

There are **five** stages to the simulation life cycle in ``vivarium``. We will go through each of them as they appear
from the entry point of ``simulate run``.

1. Initialization
    First, ``vivarium`` will set up the results directory as described above, creating the necessary subdirectories. Then,
    the core framework of ``vivarium`` takes over. It parses your model specification, setting up a PluginManager with the
    plugins provided; creating a list of the provided components; and creating a :class:`vivarium.engine.SimulationContext`
    to manage the running of the simulation. At this point, your model specification has been fully parsed and all
    managers and top-level components have been initialized.

2. Setup
    In this stage, the framework moves to setting up the components. For each top level component, the framework applies
    any defaults of the component. Next, it calls ``setup`` on each component. At this stage, components may spawn
    additional components, so this process continues until all components are setup. Setting up components may involve
    loading data, registering or getting pipelines, creating lookup tables, registering population initializers, getting
    randomness streams, etc. The specifics of this are determined by the :func:`setup` method on each component - the
    framework itself simply calls that method.

3. Initialization of the Base Population
    It's not until this stage that the framework actually generates the base population for the simulation. Here, the
    framework rewinds the simulation clock one time step and generates the population with ages smeared between the
    simulation start time and that start time minus one time step. Note that this rewinding of the clock is purely what
    it sounds like - there is no concept of a time step being taken here. Instead, the clock is literally reset back the
    duration of one time step. Once the simulant population is generated, the clock is reset to the simulation start
    time, again by changing the clock time only without any time step being taken.

4. Event Loop
    At this stage, all the preparation work has been completed and the framework begins to move through the simulation.
    This occurs as an event loop. The framework emits a series of events for each time step: *time_step__prepare*,
    *time_step*, *time_step__cleanup*, *collect_metrics*. By listening for these events, individual components can
    perform actions, including manipulating simulants. In this way does the engine drive the simulation forward.

    .. note::

        Note that we have multiple sources of time during this process. The :class:`vivarium.engine.SimulationContext`
        itself holds onto a clock. This simulation clock is the actual time in the simulation. Events (including e.g.,
        *time_step*) come with a time as well. This time is the time at the start of the next time step, that is, the
        time when any changes made during the loop will happen.

5. Finalization
    The final stage in the simulation life cycle is fittingly enough, finalization. At this stage, the *simulation_end*
    event is emitted to signal that the event loop has finished and the state table is final. At this point, outputs
    should be computed.

These five stages together make up the life cycle of a ``vivarium`` simulation. ``simulate run`` provides your entry
into this life cycle. Supply a model specification to ``simulate run`` and the simulation engine will use it to define
the simulation that progresses through each of these stages.
