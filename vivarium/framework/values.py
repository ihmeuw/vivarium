"""The mutable value system
"""
from collections import defaultdict

import pandas as pd

from vivarium import config, VivariumError

from .util import marker_factory, from_yearly

produces_value = marker_factory('value_system__produces')
produces_value.__doc__ = """Mark a function as the producer of the named value."""
modifies_value = marker_factory('value_system__modifies', with_priority=True)
modifies_value.__doc__ = """Mark a function as a mutator of the named value.
Mutators will be evaluated in `priority` order (lower values happen first)"""


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


def rescale_post_processor(a):
    """Assumes that the value is an annual rate and rescales it to the
    current time step.
    """
    time_step = config.simulation_parameters.time_step
    return from_yearly(a, pd.Timedelta(time_step, unit='D'))


def joint_value_post_processor(a):
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

def _dummy_source(*args, **kwargs):
    raise DynamicValueError('No source for value.')

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
        self.source = _dummy_source
        self.mutators = [[] for i in range(10)]
        self.combiner = combiner
        self.post_processor = post_processor
        self.configured = False

    def __call__(self, *args, skip_post_processor=False, **kwargs):
        value = self.source(*args, **kwargs)
        for priority_bucket in self.mutators:
            for mutator in priority_bucket:
                value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor and not skip_post_processor:
            return self.post_processor(value)

        return value

    def __repr__(self):
        mutators = {i: [m.__name__ for m in b] for i, b in enumerate(self.mutators)}
        post_processor = self.post_processor.__name__ if self.post_processor else 'None'
        source = self.source.__name__ if hasattr(self.source, __name__) else self.source.__class__.__name__

        return ("Pipeline(\nsource= {},\nmutators= {},\n".format(source, mutators)
                + "combiner= {},\n post_processor= {},\n".format(self.combiner.__name__, post_processor)
                + "configured = {})".format(self.configured))


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
        # TODO : This method sets up value pipelines as well as getting them, which is pretty confusing when debugging.
        if name not in self._pipelines:
            for name_template, (combiner, post_processor, source) in self.__pipeline_templates.items():
                if name_template.match(name):
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
        return self.get_value(name,
                              preferred_combiner=replace_combiner,
                              preferred_post_processor=rescale_post_processor)

    def setup_components(self, components):
        for component in components:
            values_produced = [(v, component) for v in produces_value.finder(component)]
            values_produced += [(v, getattr(component, att))
                                for att in sorted(dir(component))
                                for v in produces_value.finder(getattr(component, att))]

            for name, producer in values_produced:
                self._pipelines[name].source = producer

            values_modified = [(v, component, i)
                               for priority in modifies_value.finder(component)
                               for i, v in enumerate(priority)]
            values_modified += [(v, getattr(component, att), i)
                                for att in sorted(dir(component))
                                for i, vs in enumerate(modifies_value.finder(getattr(component, att)))
                                for v in vs]

            for name, mutator, priority in values_modified:
                self._pipelines[name].mutators[priority].append(mutator)

    def __repr__(self):
        return "ValuesManager(_pipelines= {})".format(list(self._pipelines.keys()))
