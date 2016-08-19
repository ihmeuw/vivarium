from collections import defaultdict

from .util import marker_factory, resource_injector

listens_for, _listens_for = marker_factory('event_system__listens_for', with_priority=True)
emits, _set_injector = resource_injector('event_system__emits')

class Event:
    def __init__(self, time, index):
        self.time = time
        self.index = index

    def split(self, new_index):
        return Event(self.time, new_index)

class EventChannel:
    def __init__(self):
        self.listeners = [[] for i in range(10)]

    def emit(self, *args, **kwargs):
        for priority_bucket in self.listeners:
            for listener in priority_bucket:
                listener(*args, **kwargs)


class EventManager:
    def __init__(self):
        self.__event_types = defaultdict(EventChannel)

    def get_emitter(self, label):
        return self.__event_types[label].emit

    def emitter_injector(self, args, label):
        return list(args) + [self.__event_types[label]]

    def setup_components(self, components):
        _set_injector(self.emitter_injector)
        for component in components:
            listeners = [(v, component, i) for priority in _listens_for(component) for i,v in enumerate(priority)]
            listeners += [(v, getattr(component, att), i) for att in sorted(dir(component)) for priority in _listens_for(getattr(component, att)) for i,v in enumerate(priority)]

            for event, listener, priority in listeners:
                self.__event_types[event].listeners[priority].append(listener)
