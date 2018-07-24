"""The event framework"""
from collections import defaultdict
from typing import Callable


class Event:
    """An Event object represents the context of an event. It is possible to subclass Event
    to add information for more specialized cases like vivarium.framework.population.PopulationEvent

    Attributes
    ----------
    time  : pandas.Timestamp
            The simulation time at which this event was emitted.
    index : pandas.Index
            An index into the population table containing all simulants effected by this event.
    """

    def __init__(self, index, user_data=None):
        self.index = index
        self.user_data = user_data if user_data is not None else {}
        self.time = None
        self.step_size = None

    def split(self, new_index):
        """Create a new event which is a copy of this one but with a new index.
        """
        new_event = Event(new_index, self.user_data)
        new_event.time = self.time
        new_event.step_size = self.step_size
        return new_event

    def __repr__(self):
        return "Event(user_data={}, time={}, step_size={})".format(self.user_data, self.time, self.step_size)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


class _EventChannel:
    def __init__(self, manager):
        self.manager = manager

        self.listeners = [[] for _ in range(10)]

    def emit(self, event):
        """Notifies all listeners to this channel that an event has occurred.

        Parameters
        ----------
        event : Event
            The event to be emitted.
        """
        if hasattr(event, 'time'):
            event.step_size = self.manager.step_size()
            event.time = self.manager.clock() + self.manager.step_size()

        for priority_bucket in self.listeners:
            for listener in sorted(priority_bucket, key=lambda x: x.__name__):
                listener(event)
        return event

    def __repr__(self):
        return "_EventChannel(listeners: {})".format([listener for bucket in self.listeners for listener in bucket])


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

    def setup(self, builder):
        """Performs this components simulation setup.

        Parameters
        ----------
        builder : vivarium.framework.engine.Builder
            Object giving access to core framework functionality.
        """
        self.clock = builder.time.clock()
        self.step_size = builder.time.step_size()

    def get_emitter(self, name):
        """Get an emitter function for the named event

        Parameters
        ----------
        name : str
               The name of the event

        Returns
        -------
        callable
            A function that accepts an Event object and distributes
            it to all listeners for this event.
        """
        return self._event_types[name].emit

    def register_listener(self, name, listener, priority=5):
        """Registers a new listener to the named event.

        Parameters
        ----------
        name : str
            The name of the event.
        listener : Callable
            The consumer of the named event.
        priority : int
            Number in range(10) used to assign the ordering in which listeners process the event.
        """
        self._event_types[name].listeners[priority].append(listener)

    def get_listeners(self, name):
        channel = self._event_types[name]
        return {priority: listeners for priority, listeners in enumerate(channel.listeners) if listeners}

    def list_events(self):
        """List all event names known to the event system

        Returns
        -------
        List[Event]
            A list of all known event names

        Notes
        -----
        This value can change after setup if components dynamically create new event labels.
        """
        return list(self._event_types.keys())

    def __contains__(self, item):
        return item in self._event_types

    def __repr__(self):
        return "EventManager(event_types: {})".format(self._event_types.keys())


class EventInterface:

    def __init__(self, event_manager: EventManager):
        self._event_manager = event_manager

    def get_emitter(self, name: str) -> Callable[[Event], Event]:
        """Gets and emitter for a named event.

        Parameters
        ----------
        name :
            The name of the event he requested emitter will emit.
            Users may provide their own named events by requesting an emitter with this function,
            but should do so with caution as it makes time much more difficult to think about.

        Returns
        -------
            An emitter for the named event. The emitter should be called by the requesting component
            at the appropriate point in the simulation lifecycle.
        """
        return self._event_manager.get_emitter(name)

    def register_listener(self, name: str, listener: Callable[[Event], None], priority: int=5) -> None:
        """Registers a callable as a listener to a events with the given name.

        The listening callable will be called with a named ``Event`` as it's only argument any time the
        event emitter is invoked from somewhere in the simulation.

        The framework creates the following events and emits them at different points in the simulation:
            At the end of the setup phase: ``post_setup``
            Every time step: ``time_step__prepare``, ``time_step``, ``time_step__cleanup``, ``collect_metrics``
            At simulation end: ``simulation_end``

        Parameters
        ----------
        name :
            The name of the event to listen for.
        listener :
            The callable to be invoked any time an ``Event`` with the given name is emitted.
        priority : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            An indication of the order in which event listeners should be called. Listeners with
            smaller priority values will be called earlier. Listeners with the same priority have
            no guaranteed ordering.  This feature should be avoided if possible. Components should
            strive to obey the Markov property as they transform the state table (the state of the simulation
            at the beginning of the next time step should only depend on the current state of the system).
        """
        self._event_manager.register_listener(name, listener, priority)
