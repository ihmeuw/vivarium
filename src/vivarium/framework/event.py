"""
============================
The Vivarium Event Framework
============================

``vivarium`` constructs and manages the flow of :ref:`time <time_concept>`
through the emission of regularly scheduled events. The tools in this module
manage the relationships between event emitters and listeners and provide
an interface for user :ref:`components <components_concept>` to register
themselves as emitters or listeners to particular events.

The :class:`EventManager` maintains a mapping between event types and channels.
Each event type (and event types must be unique so event type is equivalent to
event name, e.g., ``time_step_prepare``) corresponds to an
:class:`_EventChannel`, which tracks listeners to that event in prioritized
levels and passes on the event to those listeners when emitted.

The :class:`EventInterface` is exposed off the :ref:`builder <builder_concept>`
and provides two methods: :func:`get_emitter <EventInterface.get_emitter>`,
which returns a callable emitter for the given event type and
:func:`register_listener <EventInterface.register_listener>`, which adds the
given listener to the event channel for the given event. This is the only part
of the event framework with which client code should interact.

For more information, see the associated event
:ref:`concept note <event_concept>`.

"""
from collections import defaultdict
from typing import Callable, Dict, List

import pandas as pd


class Event:
    """An Event object represents the context of an event.

    Events themselves are just a bundle of data.  They must be emitted
    along an :class:`_EventChannel` in order for other simulation components
    to respond to them.

    Attributes
    ----------
    index : pandas.Index
        An index into the population table containing all simulants
        affected by this event.
    time  : pandas.Timestamp
        The simulation time at which this event will resolve. The current
        simulation size plus the current time step size.
    step_size : pandas.Timedelta
        The current step size at the time of the event.
    user_data : dict
        Any additional data provided by the user about the event.

    """
    def __init__(self, index: pd.Index, user_data: Dict = None):
        self.index = index
        self.user_data = user_data if user_data is not None else {}
        self.time = None
        self.step_size = None

    def split(self, new_index: pd.Index) -> 'Event':
        """Create a copy of this event with a new index.

        This function should be used to emit an event in a new
        :class:`_EventChannel` in response to an event emitted from a
        different channel.

        Parameters
        ----------
        new_index
            An index into the population table containing all simulants
            affected by this event.

        Returns
        -------
            The new event.

        """
        new_event = Event(new_index, self.user_data)
        new_event.time = self.time
        new_event.step_size = self.step_size
        return new_event

    def __repr__(self):
        return f"Event(user_data={self.user_data}, time={self.time}, step_size={self.step_size})"

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class _EventChannel:
    """A named subscription channel that passes events to event listeners."""
    def __init__(self, manager):
        self.manager = manager
        self.listeners = [[] for _ in range(10)]

    def emit(self, event: Event) -> Event:
        """Notifies all listeners to this channel that an event has occurred.

        Events are emitted to listeners in order of priority (with order 0 being
        first and order 9 last), with no ordering within a particular priority
        level guaranteed.

        Parameters
        ----------
        event
            The event to be emitted.

        """
        if hasattr(event, 'time'):
            event.step_size = self.manager.step_size()
            event.time = self.manager.clock() + self.manager.step_size()

        for priority_bucket in self.listeners:
            for listener in priority_bucket:
                listener(event)
        return event

    def __repr__(self):
        return f"_EventChannel(listeners: {[listener for bucket in self.listeners for listener in bucket]})"


class EventManager:
    """The configuration for the event system.

    Notes
    -----
    Client code should never need to interact with this class
    except through the decorators in this module and the emitter
    function exposed on the builder during the setup phase.

    """

    def __init__(self):
        self._event_types = defaultdict(lambda: _EventChannel(self))

    @property
    def name(self):
        """The name of this component."""
        return "event_manager"

    def setup(self, builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder
            Object giving access to core framework functionality.

        """
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def get_emitter(self, name: str) -> Callable:
        """Get an emitter function for the named event.

        Parameters
        ----------
        name
            The name of the event.

        Returns
        -------
            A function that accepts an Event object and distributes
            it to all listeners for this event.

        """
        return self._event_types[name].emit

    def register_listener(self, name: str, listener: Callable, priority: int = 5):
        """Registers a new listener to the named event.

        Parameters
        ----------
        name
            The name of the event.
        listener
            The consumer of the named event.
        priority
            Number in range(10) used to assign the ordering in which listeners
            process the event.
        """
        self._event_types[name].listeners[priority].append(listener)

    def get_listeners(self, name: str) -> Dict[int, List[Callable]]:
        """Get  all listeners registered for the named event.

        Parameters
        ----------
        name
            The name of the event.

        Returns
        -------
            A dictionary that maps each priority level of the named event's
            listeners to a list of listeners at that level.
        """
        channel = self._event_types[name]
        return {priority: listeners for priority, listeners in enumerate(channel.listeners) if listeners}

    def list_events(self) -> List[Event]:
        """List all event names known to the event system.

        Returns
        -------
            A list of all known event names.

        Notes
        -----
        This value can change after setup if components dynamically create
        new event labels.

        """
        return list(self._event_types.keys())

    def __contains__(self, item):
        return item in self._event_types

    def __repr__(self):
        return "EventManager()"


class EventInterface:
    """The public interface for the event system."""

    def __init__(self, event_manager: EventManager):
        self._event_manager = event_manager

    def get_emitter(self, name: str) -> Callable[[Event], Event]:
        """Gets an emitter for a named event.

        Parameters
        ----------
        name
            The name of the event the requested emitter will emit.
            Users may provide their own named events by requesting an emitter
            with this function, but should do so with caution as it makes time
            much more difficult to think about.

        Returns
        -------
            An emitter for the named event. The emitter should be called by
            the requesting component at the appropriate point in the simulation
            lifecycle.

        """
        return self._event_manager.get_emitter(name)

    def register_listener(self, name: str, listener: Callable[[Event], None], priority: int = 5) -> None:
        """Registers a callable as a listener to a events with the given name.

        The listening callable will be called with a named ``Event`` as its
        only argument any time the event emitter is invoked from somewhere in
        the simulation.

        The framework creates the following events and emits them at different
        points in the simulation:

            - At the end of the setup phase: ``post_setup``
            - Every time step:
              - ``time_step__prepare``
              - ``time_step``
              - ``time_step__cleanup``
              - ``collect_metrics``
            - At simulation end: ``simulation_end``

        Parameters
        ----------
        name
            The name of the event to listen for.
        listener
            The callable to be invoked any time an ``Event`` with the given
            name is emitted.
        priority : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            An indication of the order in which event listeners should be
            called. Listeners with smaller priority values will be called
            earlier. Listeners with the same priority have no guaranteed
            ordering.  This feature should be avoided if possible. Components
            should strive to obey the Markov property as they transform the
            state table (the state of the simulation at the beginning of the
            next time step should only depend on the current state of the
            system).

        """
        self._event_manager.register_listener(name, listener, priority)
