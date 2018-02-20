"""The mutable value system"""
from collections import defaultdict

import pandas as pd

from vivarium import VivariumError


class DynamicValueError(VivariumError):
    """Indicates that a value was invoked without being properly configured.
    i.e. no source specified.
    """
    pass


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


def rescale_post_processor(a, time_step):
    """Assumes that the value is an annual rate and rescales it to the
    current time step.
    """
    return from_yearly(a, pd.Timedelta(time_step, unit='D'))


def joint_value_post_processor(a, _):
    """The final step in calculating joint values like disability weights.
    If the combiner is list_combiner then the effective formula is:
    :math:`value(args) = 1 -  \prod_{i=1}^{mutator count} 1-mutator_{i}(args)`

    Parameters
    ----------
    a : list
        a is a list of series, indexed on the population. Each series
        corresponds to a different value in the pipeline and each row
        in a series contains a value that applies to a specific simulant.
    """
    # if there is only one value, return the value
    if len(a) == 1:
        return a[0]

    # if there are multiple values, calculate the joint value
    product = 1
    for v in a:
        new_value = (1-v)
        product = product * new_value
    joint_value = 1 - product
    return joint_value


class _Pipeline:
    """A single mutable value."""

    def __init__(self):
        self.name = None
        self.source = None
        self.mutators = [[] for _ in range(10)]
        self.combiner = None
        self.post_processor = None
        self.manager = None
        self.configured = False

    def __call__(self, *args, skip_post_processor=False, **kwargs):
        if not self.source:
            raise DynamicValueError(f"The dynamic value pipeline for {self.name} has no source. This likely means"
                                    f"you are attempting to modify a value that hasn't been created.")
        elif not self.configured:
            raise DynamicValueError(f"The dynamic value pipeline for {self.name} has a source but "
                                    f"has not been configured.  You've done a weird thing to get in this state.")

        value = self.source(*args, **kwargs)
        for priority_bucket in self.mutators:
            for mutator in priority_bucket:
                value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor and not skip_post_processor:
            return self.post_processor(value, self.manager.step_size())

        return value

    def __repr__(self):
        return f"_Pipeline({self.name})"


class ValuesManager:
    """The configuration of the dynamic values system.

    Notes
    -----
    Client code should never need to interact with this class
    except through the dynamic value and rate constructors exposed
    via the builder during the setup phase.
    """

    def __init__(self):
        self._pipelines = defaultdict(_Pipeline)

    def setup(self, builder):
        self.step_size = builder.step_size()

    def register_value_modifier(self, value_name, modifier, priority=5):
        pipeline = self._pipelines[value_name]
        pipeline.mutators[priority].append(modifier)

    def register_value_producer(self, value_name, source=None,
                                preferred_combiner=replace_combiner, preferred_post_processor=None):
        pipeline = self._pipelines[value_name]
        pipeline.name = value_name
        pipeline.source = source
        pipeline.combiner = preferred_combiner
        pipeline.post_processor = preferred_post_processor
        pipeline.manager = self
        pipeline.configured = True
        return pipeline

    def register_rate_producer(self, rate_name, source=None):
        return self.register_value_producer(rate_name, source, preferred_post_processor=rescale_post_processor)

    def get_value(self, name):
        if name not in self._pipelines:
            raise DynamicValueError(f"The dynamic value {name} has not been registered with the value system.")
        return self._pipelines[name]

    def get_rate(self, name):
        if name not in self._pipelines:
            raise DynamicValueError(f"The dynamic rate {name} has not been registered with the value system.")
        return self._pipelines[name]

    def setup_components(self, components):
        for component in components:
            values_produced = [(v, component) for v in produces_value.finder(component)]
            values_produced += [(v, getattr(component, att))
                                for att in sorted(dir(component)) if callable(getattr(component, att))
                                for v in produces_value.finder(getattr(component, att))]

            for name, producer in values_produced:
                self._pipelines[name].source = producer

            values_modified = [(v, component, i)
                               for priority in modifies_value.finder(component)
                               for i, v in enumerate(priority)]
            values_modified += [(v, getattr(component, att), i)
                                for att in sorted(dir(component)) if callable(getattr(component, att))
                                for i, vs in enumerate(modifies_value.finder(getattr(component, att)))
                                for v in vs]

            for name, mutator, priority in values_modified:
                self._pipelines[name].mutators[priority].append(mutator)

    def __contains__(self, item):
        return item in self._pipelines

    def __iter__(self):
        return iter(self._pipelines)

    def keys(self):
        return self._pipelines.keys()

    def items(self):
        return self._pipelines.items()

    def __repr__(self):
        return "ValuesManager(_pipelines= {})".format(list(self._pipelines.keys()))
