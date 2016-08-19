from collections import defaultdict
import re
from datetime import timedelta

from ceam import config

from ceam.util import from_yearly
from .util import marker_factory
from .event import listens_for

produces_value, _values_produced = marker_factory('value_system__produces')
modifies_value, _values_modified = marker_factory('value_system__modifies', with_priority=True)

def replace_combiner(value, mutator, *args, **kwargs):
    args = list(args) + [value]
    return mutator(*args, **kwargs)

def joint_value_combiner(value, mutator, *args, **kwargs):
    new_value = mutator(*args, **kwargs)
    return value * (1-new_value)

def rescale_post_processor(a):
    time_step = config.getfloat('simulation_parameters', 'time_step')
    return from_yearly(a, timedelta(days=time_step))

def joint_value_post_processor(a):
    return 1-a

class Pipeline:
    def __init__(self, combiner=replace_combiner, post_processor=rescale_post_processor):
        self.source = None
        self.mutators = [[] for i in range(10)]
        self.combiner = combiner
        self.post_processor = post_processor

    def __call__(self, *args, **kwargs):
        value = self.source(*args, **kwargs)
        for priority_bucket in self.mutators:
            for mutator in priority_bucket:
                value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor:
            return self.post_processor(value)
        else:
            return value

class ValuesManager:
    def __init__(self):
        self._pipelines = defaultdict(Pipeline)
        self.__pipeline_templates = {}

    def mutator(self, mutator, label, priority=5):
        pipeline = self.get_pipeline(label)
        pipeline.mutators[priority].append(mutator)

    def get_pipeline(self, label):
        if label not in self._pipelines:
            for label_template, (combiner, post_processor, source) in self.__pipeline_templates.items():
                if label_template.match(label):
                    self._pipelines[label] = Pipeline(combiner=combiner, post_processor=post_processor)
                    if source:
                        self._pipelines[label].source = source
        return self._pipelines[label]

    def declare_pipeline(self, label, combiner=replace_combiner, post_processor=rescale_post_processor, source=None):
        if hasattr(label, 'match'):
            # This is a compiled regular expression
            self.__pipeline_templates[label] = (combiner, post_processor, source)
        else:
            self._pipelines[label] = Pipeline(combiner=combiner, post_processor=post_processor)
            if source:
                self._pipelines[label].source = source

    def setup_components(self, components):
        for component in components:
            values_produced = [(v, component) for v in _values_produced(component)]
            values_produced += [(v, getattr(component, att)) for att in sorted(dir(component)) for v in _values_produced(getattr(component, att))]

            for value, producer in values_produced:
                pipeline = self.get_pipeline(value)
                pipeline.source = producer

            values_modified = [(v, component, i) for priority in _values_modified(component) for i,v in enumerate(priority)]
            values_modified += [(v, getattr(component, att), i) for att in sorted(dir(component)) for i,vs in enumerate(_values_modified(getattr(component, att))) for v in vs]

            for value, mutator, priority in values_modified:
                pipeline = self.get_pipeline(value)
                pipeline.mutators[priority].append(mutator)
