# ~/ceam/engine.py

from collections import defaultdict

class Event(object):
    def __init__(self, label):
        self.label = label

class PopulationEvent(Event):
    def __init__(self, label, affected_population):
        super(PopulationEvent, self).__init__(label)
        self.affected_population = affected_population

class EventHandler(object):
    def __init__(self):
        super(EventHandler, self).__init__()
        self._listeners_store = [defaultdict(set) for _ in range(10)]

    def _listeners(self, label):
        listeners = []
        for priority_level in self._listeners_store:
            listeners += priority_level[label]
            if label is not None:
                listeners += priority_level[None]
        return listeners

    def register_event_listener(self, listener, label=None, priority=5):
        assert callable(listener), "Listener must be callable"
        assert priority in range(10), "Priority must be 0-9"

        self._listeners_store[priority][label].add(listener)

    def deregister_event_listener(self, listener, label=None):
        for priority_level in self._listeners_store:
            if label in priority_level:
                if listener in priority_level[label]:
                    priority_level[label].remove(listener)

    def emit_event(self, event):
        for listener in self._listeners(event.label):
            listener(event)

#End.
