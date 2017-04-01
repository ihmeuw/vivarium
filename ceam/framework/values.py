"""The mutable value system
"""

from collections import defaultdict
import re
from datetime import timedelta

import pandas as pd

from ceam import config, CEAMError

from .util import marker_factory, from_yearly
from .event import listens_for

produces_value = marker_factory('value_system__produces')
produces_value.__doc__ = """Mark a function as the producer of the named value."""
modifies_value = marker_factory('value_system__modifies', with_priority=True)
modifies_value.__doc__ = """Mark a function as a mutator of the named value.
Mutators will be evaluated in `priority` order (lower values happen first)"""

class DynamicValueError(CEAMError):
    """Indicates that a value was invoked without being properly configured.
    i.e. no source specified.
    """
    pass

class NullValue:
    """An empty value that can carry an index. Used by joint value pipelines
    as their source value.
    """
    def __init__(self, index):
        self.index = index

def replace_combiner(value, mutator, *args, **kwargs):
    """Replaces the output of the source or mutator with the output
    of the subsequent mutator. This is the default combiner.
    """
    args = list(args) + [value]
    return mutator(*args, **kwargs)

def set_combiner(value, mutator, *args, **kwargs):
    """Expects the output of the source to be a set to which
    the result of each mutator is added.
    """
    value.add(mutator(*args, **kwargs))
    return value

def list_combiner(value, mutator, *args, **kwargs):
    """Expects the output of the source to be a list to which
    the result of each mutator is appended.
    """
    value.append(mutator(*args, **kwargs))
    return value

def joint_value_combiner(value, mutator, *args, **kwargs):
    """Combines the output of the source or previous mutator with the output of
    the subsequent mutator using this formula: a * (1-b) Used for PAFs
    and disability weights.

    Notes
    -----
    If the input value is NullValue then the output of the mutator is returned
    unmodified. This is useful when there is no meaningful source, as when
    calculating PAFs.
    """
    new_value = mutator(*args, **kwargs)
    if isinstance(value, NullValue):
        return 1 - new_value
    else:
        return value * (1-new_value)

def rescale_post_processor(a):
    """Assumes that the value is an annual rate and rescales it to the 
    current time step.
    """
    time_step = config.simulation_parameters.time_step
    return from_yearly(a, timedelta(days=time_step))

def joint_value_post_processor(a):
    """The final step in calculating joint values like PAFs. If the combiner
    is joint_value_combiner then the effective formula is:
    :math:`value(args) = 1 -  \prod_{i=1}^{mutator count} 1-mutator_{i}(args)`
    """
    if isinstance(a, NullValue):
        return pd.Series(1, index=a.index)
    else:
        return 1-a

class Pipeline:
    """A single mutable value.

    Attributes
    ----------
    source         : callable
                     The function which generates the base form of this value.
    mutators       : [[callable]]
                     A list of priority buckets containing functions that mutate this value.
    combiner       : callable
                     The function to use when combining the results of subsequent mutators.
    post_processor : callable
                     A function which processes the output of the last mutator. If None, no post-processing is done.
    """

    def __init__(self, combiner=replace_combiner, post_processor=None):
        self.source = None
        self.mutators = [[] for i in range(10)]
        self.combiner = combiner
        self.post_processor = post_processor
        self.configured = False

    def __call__(self, *args, skip_post_processor=False, **kwargs):
        if self.source is None:
            raise DynamicValueError('No source for value.')
        value = self.source(*args, **kwargs)
        for priority_bucket in self.mutators:
            for mutator in priority_bucket:
                value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor and not skip_post_processor:
            return self.post_processor(value)
        else:
            return value

class ValuesManager:
    """The configuration of the dynamic values system.

    Notes
    -----
    Client code should never need to interact with this class
    except through the dynamic value and rate constructors exposed
    via the builder during the setup phase.
    """

    def __init__(self):
        self._pipelines = defaultdict(Pipeline)
        self.__pipeline_templates = {}

    def mutator(self, mutator, value_name, priority=5):
        pipeline = self.get_value(value_name)
        pipeline.mutators[priority].append(mutator)

    def get_value(self, name, preferred_combiner=None, preferred_post_processor=None):
        """Get a reference to the named dynamic value which can be called to get it's effective value.

        Parameters
        ----------
        name                     : str
                                   The name of the value
        preferred_combiner       : callable
                                   The combiner to use if the value is not already configured
        preferred_post_processor : callable
                                   The post-processor to use if the value is not already configured
        """
        if name not in self._pipelines:
            found_template = False
            for name_template, (combiner, post_processor, source) in self.__pipeline_templates.items():
                if name_template.match(name):
                    found_template = True
                    self._pipelines[name] = Pipeline(combiner=combiner, post_processor=post_processor)
                    if source:
                        self._pipelines[name].source = source
                    self._pipelines[name].configured = True

        if not self._pipelines[name].configured:
            if preferred_combiner:
                self._pipelines[name].combiner = preferred_combiner
                self._pipelines[name].configured = True
            if preferred_post_processor:
                self._pipelines[name].post_processor = preferred_post_processor
                self._pipelines[name].configured = True

        return self._pipelines[name]

    def get_rate(self, name):
        """Get a reference to the named dynamic rate which can be called to get it's effective value.
        """
        return self.get_value(name, preferred_combiner=replace_combiner, preferred_post_processor=rescale_post_processor)

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
            values_produced = [(v, component) for v in produces_value.finder(component)]
            values_produced += [(v, getattr(component, att)) for att in sorted(dir(component)) for v in produces_value.finder(getattr(component, att))]

            for name, producer in values_produced:
                self._pipelines[name].source = producer

            values_modified = [(v, component, i) for priority in modifies_value.finder(component) for i,v in enumerate(priority)]
            values_modified += [(v, getattr(component, att), i) for att in sorted(dir(component)) for i,vs in enumerate(modifies_value.finder(getattr(component, att))) for v in vs]

            for name, mutator, priority in values_modified:
                self._pipelines[name].mutators[priority].append(mutator)
