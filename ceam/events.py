# ~/ceam/ceam/events.py

from collections import defaultdict


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

    def emit_event(self, label, mask, simulation):
        for listener in self._listeners(label):                         # Function call, not reference to instance variable.
            listener(label, mask.copy(), simulation)


#End.
