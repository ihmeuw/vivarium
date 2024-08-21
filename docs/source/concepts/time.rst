.. _time_concept:

=====================================
Thinking about Time in the Simulation
=====================================

.. contents::
   :depth: 2
   :local:
   :backlinks: none

The Simulation Clock
--------------------
The :class:`SimulationClock <vivarium.framework.time.SimulationClock>` plugin manages the progression of time throughout the simulation. 
Fundamentally, that means it keeps track of the current time (beginning at the *start time*), provides
a mechanism to advance the simulation time by some duration (the *step size*), and determines when 
the simulation is complete via a configured *end time*. The simplest
implementation of a Clock is the :class:`SimpleClock <vivarium.framework.time.SimpleClock>` object, which is little more
than an integer counter that is incremented by a fixed step size until it reaches the
end time. Modeling real-world events is often dependent on data that are tied to particular years or rates, where the 
desired step size is not necessarily known in advance. Therefore, it is common to use the :class:`DateTimeClock <vivarium.framework.time.DateTimeClock>`,
which uses datetime-like objects (specifically :class:`~pandas.Timestamp` and :class:`~pandas.Timedelta`) as the temporal units. The DateTimeClock
can more easily facilitate the conversion rates to particular increments of time.

Event Times
-----------
Discrete time simulations assume that all changes to a simulant's state vector happen at the 
end of the time step, that is, the current clock time *plus* the step size. :mod:`vivarium` explicates this important distinction 
and labels this quantity the *event time*. `Events <events_concept>` that correspond to (potential) state changes are mediated through the
:class:`Event Manager <vivarium.framework.event.EventManager>`, which propagates events to :ref:`components <components_concept>` subscribed to them during particuar phases of the simulation lifecycle.
The Event Manager uses the event time when calculating time-related outcomes, for example, age- or year-dependent rates of morbidity and mortality.

Time Interface
--------------
The Time plugin provides, via the :ref:`Builder <builder_concept>`, an :class:`interface <vivarium.framework.time.TimeInterface>` to access several clock methods that might be needed
by other managers or components. In particular, components can access the current time and step size (and, implicitly, the event time).

Individual Clocks
-----------------
:mod:`vivarium` also allows one to update simulants asynchronously with different frequencies depending on their state information.
For example, a component that simulates the progression of a disease might need to update the state of each
simulant more frequently when infected than when in remission. The basic method is to give each simulant its own distinct clock time and step size instead of one global clock. 
A simulant's *next event time*, that is, the sum of its clock time and step size, is when it is scheduled to be updated.
Currently, the :mod:`vivarium` still incorporates a global clock, which determines the start, end, and minimal step size of the simulation. The minimum step
size is the smallest value that a simulant's step size can take, and therefore determines the minimum duration by which the simulation can advance in a single iteration.
However, global step size changes from iteration to iteration and can be larger than the minimum step size. In each iteration of the simulation, the global clock is advanced to the earliest time in which some simulant is scheduled to be updated. 
Simulants that are not scheduled to be updated in a particular iteration are simply excluded from the relevant events as propagated by the Event Manager. 
In effect, if there are no simulants to be updated in a duration comprising several minimum timesteps, those "minimum timesteps" are skipped.

The Time Interface provides a method to modify a simulant's step size based on some criteria, :func:`builder.time.register_step_size_modifier() <vivarium.framework.time.TimeInterface.register_step_size_modifier>`.
If there are multiple modifiers to the same simulant simultaneously, the time manager chooses the smallest one (bounded by the global minimum step size).
If a simulant has no step modifier, it is given a default value, either the global minimum or another optionally configurable value, the *standard* step size,
in the case that we want the "background" update frequency to be larger than the minimium size.
If *no* simulants have a step modifier, then the simulation behaves as if there were no individual clocks, reverting to the global clock.