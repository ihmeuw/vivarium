from collections import defaultdict

from .util import marker_factory, resource_injector

listens_for = marker_factory('event_system__listens_for', with_priority=True)
emits = resource_injector('event_system__emits')

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

    def emitter_injector(self, func, args, kwargs, label):
        return list(args) + [self.__event_types[label]], kwargs

    def setup_components(self, components):
        emits.set_injector(self.emitter_injector)
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
        return list(self.__event_types.keys())
