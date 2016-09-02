"""The event framework
"""

from collections import defaultdict

from .util import marker_factory, resource_injector

listens_for = marker_factory('event_system__listens_for', with_priority=True)
emits = resource_injector('event_system__emits')

class Event:
    """The parent class of all events

    Attributes
    ----------
    time  : datetime.datetime
            The simulation time at which this event was emited.
    index : pandas.Index
            An index into the population table containing all simulants effected by this event.
    """
    def __init__(self, time, index):
        self.time = time
        self.index = index

    def split(self, new_index):
        """Create a new event with all the attributes of this event but a new index.
        """
        return Event(self.time, new_index)

class _EventChannel:
    def __init__(self):
        self.listeners = [[] for i in range(10)]

    def emit(self, *args, **kwargs):
        for priority_bucket in self.listeners:
            for listener in priority_bucket:
                listener(*args, **kwargs)


class EventManager:
    """Manages the configuration of the event system
    """
    def __init__(self):
        self.__event_types = defaultdict(_EventChannel)

    def get_emitter(self, label):
        return self.__event_types[label].emit

    def _emitter_injector(self, func, args, kwargs, label):
        return list(args) + [self.__event_types[label]], kwargs

    def setup_components(self, components):
        emits.set_injector(self._emitter_injector)
        for component in components:
            listeners = [(v, component, i) for priority in listens_for.finder(component) for i,v in enumerate(priority)]
            listeners += [(v, getattr(component, att), i) for att in sorted(dir(component)) for i,vs in enumerate(listens_for.finder(getattr(component, att))) for v in vs]

            for event, listener, priority in listeners:
                self.__event_types[event].listeners[priority].append(listener)

            emitters = [(v, component) for v in emits.finder(component)]
            emitters += [(v, getattr(component, att)) for att in sorted(dir(component)) for v in emits.finder(getattr(component, att))]

            # Pre-create the EventChannels for know emitters
            for (args, kwargs), emitter in emitters:
                self.get_emitter(*args, **kwargs)

    def list_events(self):
        """Return a list of all event labels known to the event system

        This value can change after setup if components dynamically create new event labels.
        """
        return list(self.__event_types.keys())
