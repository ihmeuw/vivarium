"""The event framework
"""

from collections import defaultdict

from .util import marker_factory, resource_injector

listens_for = marker_factory('event_system__listens_for', with_priority=True)
listens_for.__doc__ = """Mark a function as a listener for the named event so that
the simulation will call it when that event occurs.
"""

emits = resource_injector('event_system__emits')
emits.__doc__ = """Mark a function as an emitter for the named event. An event emitter function
which can be called to emit the named event will be injected into the functions
arguments whenever it is called.
"""


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
        self.__event_types = defaultdict(lambda: _EventChannel(self))

    def setup(self, builder):
        """Performs this components simulation setup.

        Parameters
        ----------
        builder : vivarium.framework.engine.Builder
            Object giving access to core framework functionality.
        """
        self.clock = builder.clock()
        self.step_size = builder.step_size()

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

        return self.__event_types[name].emit

    def register_listener(self, name, listener, priority=5):
        """Registers a new listener to the named event.

        Parameters
        ----------
        name : str
            The name of the event.
        listener : callable
            The consumer of the named event.
        priority : int in range(10)
            Number used to assign the ordering in which listeners process the event.
        """
        self.__event_types[name].listeners[priority].append(listener)

    def _emitter_injector(self, _, args, kwargs, label):
        return list(args) + [self.__event_types[label].emit], kwargs

    def setup_components(self, components):
        """Registers the simulation components with the event system.

        Parameters
        ----------
        components : Iterable
            The simulation components.
        """
        emits.set_injector(self._emitter_injector)
        for component in components:
            listeners = [(v, component, i)
                         for i, priority in enumerate(listens_for.finder(component))
                         for v in priority]
            listeners += [(v, getattr(component, att), i)
                          for att in sorted(dir(component)) if callable(getattr(component, att))
                          for i, vs in enumerate(listens_for.finder(getattr(component, att)))
                          for v in vs]

            for event, listener, priority in listeners:
                self.register_listener(event, listener, priority)

            emitters = [(v, component) for v in emits.finder(component)]
            emitters += [(v, getattr(component, att))
                         for att in sorted(dir(component)) if callable(getattr(component, att))
                         for v in emits.finder(getattr(component, att))]

            # Pre-create the EventChannels for known emitters
            for (args, kwargs), emitter in emitters:
                self.get_emitter(*args, **kwargs)

    def list_events(self):
        """List all event names known to the event system

        Returns
        -------
        list
            A list of all known event names

        Notes
        -----
        This value can change after setup if components dynamically create new event labels.
        """

        return list(self.__event_types.keys())

    def __contains__(self, item):
        return item in self.__event_types

    def __repr__(self):
        return "EventManager(event_types: {})".format(self.__event_types.keys())
