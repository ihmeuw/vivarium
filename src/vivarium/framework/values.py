"""The mutable value system"""
from collections import defaultdict
from typing import Callable
from types import MethodType
import logging

import pandas as pd

from vivarium import VivariumError
from .utilities import from_yearly


_log = logging.getLogger(__name__)


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

    .. math::

        value(args) = 1 -  \prod_{i=1}^{mutator count} 1-mutator_{i}(args)

    Parameters
    ----------
    a : List[pd.Series]
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


class Pipeline:
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
        self._pipelines = defaultdict(Pipeline)

    def setup(self, builder):
        self.step_size = builder.time.step_size()
        builder.event.register_listener('post_setup', self.on_post_setup)

    def on_post_setup(self, event):
        _log.debug(f"{[p for p, v in self._pipelines.items() if not v.source]}")

    def register_value_modifier(self, value_name, modifier, priority=5):
        m = modifier if isinstance(modifier, MethodType) else modifier.__call__
        _log.debug(f"Registering {str(m).split()[2]} as modifier to {value_name}")
        pipeline = self._pipelines[value_name]
        pipeline.mutators[priority].append(modifier)

    def register_value_producer(self, value_name, source=None,
                                preferred_combiner=replace_combiner, preferred_post_processor=None):
        _log.debug(f"Registering value pipeline {value_name}")
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


class ValuesInterface:
    def __init__(self, value_manager: ValuesManager):
        self._value_manager = value_manager

    def register_value_producer(self, value_name: str, source: Callable[..., pd.DataFrame]=None,
                                preferred_combiner: Callable=replace_combiner,
                                preferred_post_processor: Callable[..., pd.DataFrame]=None) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named value.

        Parameters
        ----------
        value_name :
            The name of the new dynamic value pipeline.
        source :
            A callable source for the dynamic value pipeline.
        preferred_combiner :
            A strategy for combining the source and the results of any calls to mutators in the pipeline.
            ``vivarium`` provides the strategies ``replace_combiner`` (the default), ``list_combiner``,
            and ``set_combiner`` which are importable from ``vivarium.framework.values``.  Client code
            may define additional strategies as necessary.
        preferred_post_processor :
            A strategy for processing the final output of the pipeline. ``vivarium`` provides the strategies
            ``rescale_post_processor`` and ``joint_value_post_processor`` which are importable from
            ``vivarium.framework.values``.  Client code may define additional strategies as necessary.

        Returns
        -------
        Callable
            A callable reference to the named dynamic value pipeline.
        """
        return self._value_manager.register_value_producer(value_name, source,
                                                           preferred_combiner,
                                                           preferred_post_processor)

    def register_rate_producer(self, rate_name: str, source: Callable[..., pd.DataFrame]=None) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named rate.

        This is a convenience wrapper around ``register_value_producer`` that makes sure
        rate data is appropriately scaled to the size of the simulation time step.
        It is equivalent to ``register_value_producer(value_name, source,
        preferred_combiner=replace_combiner, preferred_post_processor=rescale_post_processor)``

        Parameters
        ----------
        rate_name :
            The name of the new dynamic rate pipeline.
        source :
            A callable source for the dynamic rate pipeline.

        Returns
        -------
        Callable
            A callable reference to the named dynamic rate pipeline.
        """
        return self._value_manager.register_rate_producer(rate_name, source)

    def register_value_modifier(self, value_name: str, modifier: Callable, priority: int=5):
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline when called.
            If the pipeline has a ``replace_combiner``, the modifier should accept the same
            arguments as the pipeline source with an additional last positional argument
            for the results of the previous stage in the pipeline. For the ``list_combiner`` and
            ``set_combiner`` strategies, the pipeline modifiers should have the same signature
            as the pipeline source.
        priority : {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
            An indication of the order in which pipeline modifiers should be called. Modifiers with
            smaller priority values will be called earlier in the pipeline.  Modifiers with
            the same priority have no guaranteed ordering, and so should be commutative.
        """
        self._value_manager.register_value_modifier(value_name, modifier, priority)

    def get_value(self, name):
        return self._value_manager.get_value(name)
