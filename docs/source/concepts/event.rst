.. _event_concept:

================
The Event System
================

.. contents::
   :depth: 2
   :local:
   :backlinks: none


``vivarium`` constructs and manages the flow of :ref:`time <time_concept>`
through the emission of regularly scheduled events. This event system provides
a means of coordinating across various components in a simulation.

What is an Event?
-----------------

An :class:`Event <vivarium.framework.event.Event>` is a container for a series
of attributes that provide all the necessary information to respond to the event,
including an index into the population table that contains all affected
simulants, the simulation time at which the event was emitted, the time step
size at the time of emission, and any additional user-provided data. ``vivarium``
manages these events by requiring that each type of event be unique
(e.g., there is only one ``simulation_end`` event and its type (or name - the
two are equivalent because of the uniqueness) is "simulation_end"). Individual
components are then allowed to register ``listeners`` (i.e., callable functions
to respond on the emission of an event) to each event. When an event is emitted,
all listeners that have registered themselves to that event type are called.

Lifecycle Events
----------------

The simulation engine itself is responsible for the emission of a dedicated set
of events that determine the progression of time in the simulation. Each time
step in the simulation corresponds to the emission of a set of ``time_step``
events. :ref:`Components <components_concept>` can register themselves as
listeners to these events and thus take action on each time step or other
key simulation phase.

The following table depicts the events emitted by the simulation engine itself
and the lifecycle phase when they are emitted. See the
:ref:`lifecycle concept note <lifecycle_concept>` for more information about
these phases.

.. list-table:: **Events Emitted by the Simulation Engine**
    :header-rows: 1
    :widths: 30, 40

    *   - Lifecycle Phase
        - Events
    *   - | Post-setup
        - | ``post_setup``
    *   - | Main Event Loop
        - | ``time_step_prepare``
          | ``time_step``
          | ``time_step__cleanup``
          | ``collect_metrics``
    *   - | Finalization
        - | ``simulation_end``


Interacting with Events
-----------------------

The :class:`EventInterface <vivarium.framework.event.EventInterface>` is available
off the :ref:`Builder <builder_concept>` and provides two options for interacting
with the event system:

1. :func:`register_listener <EventInterface.register_listener()>` to add a
listener to a given event to be called on emission

2. :func:`get_emitter <vivarium.framework.event.EventInterface.get_emitter()>` to
retrieve a callable emitter for a given event

Although methods for both getting emitters and registering listeners are provided,
it is strongly encouraged that only the registering listeners aspect is used.


Registering Listeners
+++++++++++++++++++++
In order to register a listener to an event to respond when that event is
emitted, we can use the :func:`register_listener <EventInterface.register_listener()>`.
The listener itself should be a callable function that takes as its only argument
the :class:`Event <vivarium.framework.event.Event>` that is emitted.

Suppose we wish to track how many simulants are affected each time step. We
could do this by creating an observer component with a ``on_time_step`` method
that we will register as a listener for the ``time_step`` event. Our component
would look something like the following:

.. code-block:: python

    class AffectedObserver:

        def setup(self, builder):
            self.affected_counts = pd.DataFrame(columns=['time_step', 'number_affected])
            builder.event.register_listener('time_step', self.on_time_step)

        def on_time_step(self, event):
            self.affected_counts.append(pd.DataFrame({'time_step': event.time, 'number_affected': len(event.index)}))

On each time step, our ``on_time_step`` method will be called and we will add
another row to our dataframe tracking the number of affected simulants.

.. note::
    Listeners are stored in priority levels when registered to an event.
    These levels (0-9) indicate which order listeners should be called when the event
    is emitted; listeners in lower priority levels will be called earlier. Within a
    priority level, there is no guarantee of order.

    **This feature should be avoided if possible.** Components should strive to obey
    the Markov property as they transform the state table: the state of the
    simulation at the beginning of the next time step should only depend on the
    current state of the system.


Emitting Events
+++++++++++++++
The :func:`get_emitter <vivarium.framework.event.EventInterface.get_emitter()>`
provides a way to get a callable emitter for a given named event. It can be used
as follows:

.. code-block:: python

    emitter = builder.event.get_emitter('my_event')

.. danger::
    Do not emit any of the simulation lifecyle events described in the table
    above. These are events that correspond to particular phases in the simulation
    and should only be emitted by the engine itself.

.. caution::
    While users may provide their own named events by requesting an emitter, this is
    not advised. Adding additional events beyond those emitted by the simulation
    engine essentially creates arbitrary GOTO statements in the simulation flow
    and makes time much more difficult to think about.



