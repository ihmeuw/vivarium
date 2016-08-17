from collections import defaultdict

from ceam.util import from_yearly
from .util import marker_factory
from .event import listens_for

produces_value, _values_produced = marker_factory('value_system__produces')
modifies_value, _values_modified = marker_factory('value_system__modifies', with_priority=True)

class Pipeline:
    def __init__(self):
        self.source = None
        self.mutators = [[] for i in range(10)]

    def __call__(self, *args, **kwargs):
        value = self.source(*args, **kwargs)
        for priority_bucket in self.mutators:
            for mutator in priority_bucket:
                new_args = list(args) + [value]
                value = mutator(*new_args, **kwargs)
        return value

class ValuesManager:
    def __init__(self):
        self.__pipelines = defaultdict(Pipeline)
        self._time_step = None

    def produces(self, label):
        def wrapper(provider):
            self.__pipelines[label].source = provider
            return provider
        return wrapper

    def modifies(self, label, priority=5):
        def wrapper(mutator):
            self.__pipelines[label].mutators[priority].append(mutator)
            return mutator
        return wrapper

    def consumes(self, label):
        def wrapper(consumer):
            def inner(*args, **kwargs):
                args = list(args) + [self.__pipelines[label]]
                return consumer(*args, **kwargs)
            return inner
        return wrapper

    def get_pipeline(self, label, rescale_to_timestep=True):
        def getter(*args, **kwargs):
            value = self.__pipelines[label](*args, **kwargs)
            if rescale_to_timestep:
                return from_yearly(value, self.time_step)
            else:
                return value
        return getter

    def setup_components(self, components):
        for component in components:
            values_produced = [(v, component) for v in _values_produced(component)]
            values_produced += [(v, getattr(component, att)) for att in sorted(dir(component)) for v in _values_produced(getattr(component, att))]

            for value, producer in values_produced:
                self.__pipelines[value].source = producer

            values_modified = [(v, component, i) for priority in _values_modified(component) for i,v in enumerate(priority)]
            values_modified += [(v, getattr(component, att), i) for att in sorted(dir(component)) for priority in _values_modified(getattr(component, att)) for i,v in enumerate(priority)]

            for value, mutator, priority in values_modified:
                self.__pipelines[value].mutators[priority].append(mutator)

    @listens_for('time_step__prepare')
    def track_time_step(self, event):
        self.time_step = event.time_step
