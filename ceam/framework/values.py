from collections import defaultdict
import re

from ceam import config

from ceam.util import from_yearly
from .util import marker_factory
from .event import listens_for

produces_value, _values_produced = marker_factory('value_system__produces')
modifies_value, _values_modified = marker_factory('value_system__modifies', with_priority=True)

def replace_combiner(a, b):
    return b

def joint_value_combiner(a, b):
    return (1-a) * (1-b)

def rescale_post_processor(a):
    time_step = config.getfloat('simulation_parameters', 'time_step')
    return from_yearly(a, timedelta(days=time_step))

def joint_value_post_processor(a):
    return 1-a

class Pipeline:
    def __init__(self, combiner=replace_combiner, post_processing=rescale_post_processor):
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
        self.__pipeline_templates = {}

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

    def get_pipeline(self, label):
        if label not in self.__pipelines:
            for label_template, (combiner, post_processing, source) in self.__pipeline_templates.items():
                if label_template.match(label):
                    self.__pipelines[label] = Pipeline(combiner=combiner, post_processing=post_processing)
                    if source:
                        self.__pipelines[label].source = source
        return self.__pipelines[label]

    def declare_pipeline(self, label, combiner=replace_combiner, post_processing=rescale_post_processor, source=None):
        if hasattr(label, 'match'):
            # This is a compiled regular expression
            self.__pipeline_templates[label] = (combiner, post_processing, source)
        else:
            self.__pipelines[label] = Pipeline(combiner=combiner, post_processing=post_processing)
            if source:
                self.__pipelines[label].source = source

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
