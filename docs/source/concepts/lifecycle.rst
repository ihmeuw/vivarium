.. _lifecycle_concept:

========================
The Simulation Lifecycle
========================

The life cycle of a :mod:`vivarium` simulation is a representation of
the different execution **states** and their transitions.  An execution state
is a clearly delineated execution period during the simulation around which
we build and enforce concrete programmatic contracts. These states
can be grouped into five important **phases**.  The phases are closely related
groups of execution states.  Contracts are not enforced directly around
phases, but they are a useful tool for thinking about the execution flow
during a simulation.

.. list-table:: **Life Cycle Phases**
   :widths: 15 65
   :header-rows: 1

   * - Phase
     - Description
   * - | :ref:`Bootstrap <lifecycle_bootstrap>`
     - | Arguments passed from the simulation
       | :ref:`entry point <entry_points_concept>` are parsed and core
       | framework systems are initialized.
   * - | :ref:`Initialization <lifecycle_initialization>`
     - | Simulation managers and :ref:`components <components_concept>` are
       | initialized and all :ref:`configuration <configuration_concept>`
       | information is loaded.
   * - | :ref:`Setup <lifecycle_setup>`
     - | Components register themselves with simulation services and
       | request access to resources by interacting with the simulation
       | :ref:`builder <builder_concept>`; the initial population is created.
   * - | :ref:`Main Loop <lifecycle_main_loop>`
     - | The core logic (as encoded in the simulation components) is executed.
   * - | :ref:`Simulation End <lifecycle_simulation_end>`
     - | The population state is finalized and results are tabulated.

The simulation itself maintains a formal representation of its internal
execution state using the tools in the :mod:`~vivarium.framework.lifecycle`
module. The tools allow the simulation to make concrete contracts about flow
of execution in simulations and to constrain the availability of certain
framework services to particular life cycle states.  This makes error handling
much more robust and allows users to more easily reason about complex
simulation models.

.. todo::

   Make a graph of service availability in the simulation.

.. contents::
   :depth: 2
   :local:
   :backlinks: none


.. _lifecycle_bootstrap:

The Bootstrap Phase
-------------------

The bootstrap and initialization phases look like an atomic operation to an
external user.  Bootstrap only exists as a separate phase because certain
operations must take place before the internal representation of the simulation
life cycle exists.

During bootstrap, all user input arguments are parsed into
an internal representation of the simulation :term:`plugins <Plugin>`,
:term:`components <Component>`, and :term:`configuration <Configuration>`.
The internal plugin representation is then parsed into the simulation managers,
the set of private and public services used to build and run simulations.
Finally, the formal representation of the simulation lifecycle is constructed
and the initialization phase begins.


.. _lifecycle_initialization:

The Initialization Phase
------------------------

The initialization phase of a :mod:`vivarium` simulation starts when the
:class:`~vivarium.framework.lifecycle.LifeCycle` is fully constructed and
ends when the ``__init__`` method of the
:class:`vivarium.framework.engine.SimulationContext` completes.

Two important things happen here:

- The internal representation of the simulation :term:`components <Component>`
  is parsed into python import paths and **all** components are instantiated
  and registered with the component manager.
- The internal representation of the :term:`configuration <Configuration>` is
  updated with all component configuration defaults.

At this point, all input arguments have been parsed, all components have been
instantiated and registered with the framework, and the configuration is
effectively complete.  In an interactive setting, this is a useful phase in
the simulation life cycle because you can add locally created components and
modify the configuration.


.. _lifecycle_setup:

The Setup Phase
---------------

The setup phase is broken down into three life cycle states.

Setup
+++++

The first state is named the same as the phase and is where the bulk of the
phases work is done. During the setup state, the simulation managers and then
the simulation components will have their ``setup`` method called with
the simulation :ref:`builder <builder_concept>` as an argument.  The
builder allows the components to request services like
:ref:`randomness <crn_concept>` or views into the
:term:`population state table <State Table>` or to register themselves
with various simulation subsystems. Setting up components may also involve
loading data, registering or getting :ref:`pipelines <values_concept>`,
creating :ref:`lookup tables <lookup_concept>`, and registering
:ref:`population initializers <population_concept>`, among other things.
The specifics of this are determined by the ``setup`` method on each component
- the framework itself simply calls that method with a
:class:`vivarium.framework.engine.Builder` object.

Post-setup
++++++++++

This is a short state that exists in the simulation mainly so that framework
:term:`managers <Plugin>` can coordinate shared state and do any necessary
cleanup.  This is the first actual :ref:`event <event_concept>` emitted by
the simulation framework.  Normal ``vivarium`` :term:`components <Component>`
should never listen for this event.  This may be enforced at a later date.

Population Initialization
+++++++++++++++++++++++++

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


.. _lifecycle_main_loop:

The Main Event Loop
-------------------

At this stage, all the preparation work has been completed and the framework
begins to move through the simulation. This occurs as an
:ref:`event loop <event_concept>`.  Like the the setup phase, the main loop
phase is broken into a series of simulation states.  The framework signals
the state transitions by emitting a series of events for each
:ref:`time step <time_concept>`:

1. *time_step__prepare*
   A state in which simulation :term:`components <Component>` can do any
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
including manipulating the :ref:`state table <population_concept>`. This
sequence of events is repeated until the simulation clock passes the
simulation end time.

.. note::

    We have multiple sources of time during this process. The
    :class:`vivarium.framework.engine.SimulationContext` itself holds onto a
    clock. This simulation clock is the actual time in the simulation. Events
    (including e.g., *time_step*) come with a time as well. This time is the
    time at the start of the next time step, that is, the time when any changes
    made during the loop will happen.


.. _lifecycle_simulation_end:

The Simulation End Phase
------------------------

The final phase in the simulation life cycle is fittingly enough,
simulation end. It is split into two states.  During the first, the
*simulation_end* :ref:`event <event_concept>` is emitted to
signal that the event loop has finished and the
:ref:`state table <population_concept>` is final. At this point, final
simulation outputs are safe to compute. The second state is *report* in
which the simulation will accumulate all final outputs and return them.
