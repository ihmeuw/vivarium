"""
=========================
The Value Pipeline System
=========================

The value pipeline system is a vital part of the :mod:`vivarium`
infrastructure. It allows for values that determine the behavior of individual
:term:`simulants <Simulant>` to be constructed across across multiple
:ref:`components <components_concept>`.

For more information about when and how you should use pipelines in your
simulations, see the value system :ref:`concept note <values_concept>`.

"""
from collections import defaultdict
from numbers import Number
from typing import Callable, List, Any, TypeVar, Union, Iterable, Tuple

from loguru import logger
import numpy as np
import pandas as pd

from vivarium.exceptions import VivariumError
from vivarium.framework.utilities import from_yearly

T = TypeVar('T')
# Supports standard algebraic operations with scalar values.
NumberLike = Union[np.ndarray, pd.Series, pd.DataFrame, Number]


class DynamicValueError(VivariumError):
    """Indicates an improperly configured value was invoked."""
    pass


def replace_combiner(value: T, mutator: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    """Replace the previous pipeline output with the output of the mutator.

    This is the default combiner.

    Parameters
    ----------
    value
        The value from the previous step in the pipeline.
    mutator
        A callable that takes in all arguments that the pipeline source takes
        in plus an additional last positional argument for the value from
        the previous stage in the pipeline.
    args, kwargs
        The same args and kwargs provided during the invocation of the
        pipeline.

    Returns
    -------
        A modified version of the input value.

    """
    args = list(args) + [value]
    return mutator(*args, **kwargs)


def list_combiner(value: List[T], mutator: Callable[..., T], *args: Any, **kwargs: Any) -> List[T]:
    """Aggregates source and mutator output into a list.

    This combiner is meant to be used with a post-processor that does some
    kind of reduce operation like summing all values in the list.

    Parameters
    ----------
    value
        A list of all values provided by the source and prior mutators in the
        pipeline.
    mutator
        A callable that returns some portion of this pipeline's final value.
    args, kwargs
        The same args and kwargs provided during the invocation of the
        pipeline.

    Returns
    -------
        The input list with new mutator portion of the pipeline value
        appended to it.

    """
    value.append(mutator(*args, **kwargs))
    return value


def rescale_post_processor(value: NumberLike, time_step: pd.Timedelta):
    """Rescales annual rates to time-step appropriate rates.

    This should only be used with a simulation using a
    :class:`~vivarium.framework.time.DateTimeClock` or another implementation
    of a clock that traffics in pandas date-time objects.

    Parameters
    ----------
    value
        Annual rates, either as a number or something we can broadcast
        multiplication over like a :mod:`numpy` array or :mod:`pandas`
        data frame.
    time_step
        A pandas time delta representing the size of the upcoming time
        step.

    Returns
    -------
        The annual rates rescaled to the size of the current time step size.

    """
    return from_yearly(value, time_step)


def union_post_processor(values: List[NumberLike], _) -> NumberLike:
    """Computes a probability on the union of the sample spaces in the values.

    Given a list of values where each value is a probability of an independent
    event, this post processor computes the probability of the union of the
    events.

    .. list-table::
       :width: 100%
       :widths: 1 3

       * - :math:`p_x`
         - Probability of event x
       * - :math:`1 - p_x`
         - Probability of not event x
       * - :math:`\prod_x(1 - p_x)`
         - Probability of not any events x
       * - :math:`1 - \prod_x(1 - p_x)`
         - Probability of any event x

    Parameters
    ----------
    values
        A list of independent proportions or probabilities, either
        as numbers or as a something we can broadcast addition and
        multiplication over.

    Returns
    -------
        The probability over the union of the sample spaces represented
        by the original probabilities.

    """
    # if there is only one value, return the value
    if len(values) == 1:
        return values[0]

    # if there are multiple values, calculate the joint value
    product = 1
    for v in values:
        new_value = (1-v)
        product = product * new_value
    joint_value = 1 - product
    return joint_value


class Pipeline:
    """A tool for building up values across several components.

    Pipelines are lazily initialized so that we don't have to put constraints
    on the order in which components are created and set up. The values manager
    will configure a pipeline (set all of its attributes) when the pipeline
    source is created.

    As long as a pipeline is not actually called in a simulation, it does not
    need a source or to be configured. This might occur when writing
    generic components that create a set of pipeline modifiers for
    values that won't be used in the particular simulation.

    Attributes
    ----------
    name
        The name of the value represented by this pipeline.
    source
        A callable source for this pipeline's value.
    mutators
        A list of callables that directly modify the pipeline source or
        contribute portions of the value.
    combiner
        A strategy for combining the source and mutator values into the
        final value represented by the pipeline.
    post_processor
        An optional final transformation to perform on the combined output of
        the source and mutators.
    manager
        A reference to the simulation values manager.

    """

    def __init__(self):
        self.name = None
        self.source = None
        self.mutators = []
        self.combiner = None
        self.post_processor = None
        self.manager = None

    def __call__(self, *args, skip_post_processor=False, **kwargs):
        """Generates the value represented by this pipeline.

        Arguments
        ---------
        skip_post_processor
            Whether we should invoke the post-processor on the combined
            source and mutator output or return without post-processing.
            This is useful when the post-processor acts as some sort of final
            unit conversion (e.g. the rescale post processor).
        args, kwargs
            Pipeline arguments.  These should be the arguments to the
            callable source of the pipeline.

        Returns
        -------
            The value represented by the pipeline.

        Raises
        ------
        DynamicValueError
            If the pipeline is invoked without a source set.

        """
        return self._call(*args, skip_post_processor=skip_post_processor, **kwargs)

    def _call(self, *args, skip_post_processor=False, **kwargs):
        if not self.source:
            raise DynamicValueError(f"The dynamic value pipeline for {self.name} has no source. This likely means"
                                    f"you are attempting to modify a value that hasn't been created.")

        value = self.source(*args, **kwargs)
        for mutator in self.mutators:
            value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor and not skip_post_processor:
            return self.post_processor(value, self.manager.step_size())

        return value

    def __repr__(self):
        return f"_Pipeline({self.name})"


class ValuesManager:
    """Manager for the dynamic value system."""

    def __init__(self):
        # Pipelines are lazily initialized by _register_value_producer
        self._pipelines = defaultdict(Pipeline)

    @property
    def name(self):
        return "values_manager"

    def setup(self, builder):
        self.step_size = builder.time.step_size()
        builder.event.register_listener('post_setup', self.on_post_setup)

        self.resources = builder.resources
        self.add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(self.register_value_producer, allow_during=['setup'])
        builder.lifecycle.add_constraint(self.register_value_modifier, allow_during=['setup'])

    def on_post_setup(self, _):
        """Finalizes dependency structure for the pipelines."""
        # Unsourced pipelines might occur when generic components register
        # modifiers to values that aren't required in a simulation.
        unsourced_pipelines = [p for p, v in self._pipelines.items() if not v.source]
        if unsourced_pipelines:
            logger.warning(f"Unsourced pipelines: {unsourced_pipelines}")

        # register_value_producer and register_value_modifier record the
        # dependency structure for the pipeline source and pipeline modifiers,
        # respectively.  We don't have enough information to record the
        # dependency structure for the pipeline itself until now, where
        # we say the pipeline value depends on its source and all its
        # modifiers.
        for name, pipe in self._pipelines.items():
            dependencies = []
            if pipe.source:
                dependencies += [f'value_source.{name}']
            else:
                dependencies += [f'missing_value_source.{name}']
            for i, m in enumerate(pipe.mutators):
                mutator_name = self._get_modifier_name(m)
                dependencies.append(f'value_modifier.{name}.{i+1}.{mutator_name}')
            self.resources.add_resources('value', [name], pipe._call, dependencies)

    def register_value_producer(self,
                                value_name: str,
                                source: Callable,
                                requires_columns: List[str] = (),
                                requires_values: List[str] = (),
                                requires_streams: List[str] = (),
                                preferred_combiner: Callable = replace_combiner,
                                preferred_post_processor: Callable = None) -> Callable:
        """Marks a ``Callable`` as the producer of a named value.

        See Also
        --------
            :meth:`ValuesInterface.register_value_producer`

        """
        pipeline = self._register_value_producer(value_name, source, preferred_combiner, preferred_post_processor)

        # The resource we add here is just the pipeline source.
        # The value will depend on the source and its modifiers, and we'll
        # declare that resource at post-setup once all sources and modifiers
        # are registered.
        dependencies = self._convert_dependencies(source, requires_columns, requires_values, requires_streams)
        self.resources.add_resources('value_source', [value_name], source, dependencies)
        self.add_constraint(pipeline._call, restrict_during=['initialization', 'setup', 'post_setup'])

        return pipeline

    def _register_value_producer(self,
                                 value_name: str,
                                 source: Callable,
                                 preferred_combiner: Callable,
                                 preferred_post_processor: Callable):
        """Configure the named value pipeline with a source, combiner, and post-processor."""
        logger.debug(f"Registering value pipeline {value_name}")
        pipeline = self._pipelines[value_name]
        if pipeline.source:
            raise DynamicValueError(f'A second component is attempting to set the source for pipeline {value_name} '
                                    f'with {source}, but it already has a source: {pipeline.source}.')
        pipeline.name = value_name
        pipeline.source = source
        pipeline.combiner = preferred_combiner
        pipeline.post_processor = preferred_post_processor
        pipeline.manager = self
        return pipeline

    def register_value_modifier(self,
                                value_name: str,
                                modifier: Callable,
                                requires_columns: List[str] = (),
                                requires_values: List[str] = (),
                                requires_streams: List[str] = ()):
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier should accept the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the pipeline modifier
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the pipeline modifier is called.
        requires_streams
            A list of the randomness streams that need to be properly sourced
            before the pipeline modifier is called.

        """
        modifier_name = self._get_modifier_name(modifier)

        pipeline = self._pipelines[value_name]  # May create a pipeline
        pipeline.mutators.append(modifier)

        name = f'{value_name}.{len(pipeline.mutators)}.{modifier_name}'
        logger.debug(f"Registering {name} as modifier to {value_name}")
        dependencies = self._convert_dependencies(modifier, requires_columns, requires_values, requires_streams)
        self.resources.add_resources('value_modifier', [name], modifier, dependencies)

    def get_value(self, name):
        """Retrieve the pipeline representing the named value.

        Parameters
        ----------
        name
            Name of the pipeline to return.

        Returns
        -------
            A callable reference to the named pipeline.  The pipeline arguments
            should be identical to the arguments to the pipeline source
            (frequently just a :class:`pandas.Index` representing the
            simulants).

        """
        return self._pipelines[name]  # May create a pipeline.

    @staticmethod
    def _convert_dependencies(func, requires_columns, requires_values, requires_streams):
        # If declaring a pipeline as a value source or modifier, columns and
        # streams are optional since the pipeline itself will have all the
        # appropriate dependencies. In any situation, make sure we don't have
        # provide the pipeline function to source/modifier as well as
        # explicitly stating the pipeline name in 'requires_values'.
        if isinstance(func, Pipeline):
            dependencies = [f'value.{func.name}']
        else:
            dependencies = ([f'column.{name}' for name in requires_columns]
                            + [f'value.{name}' for name in requires_values]
                            + [f'stream.{name}' for name in requires_streams])
        return dependencies

    @staticmethod
    def _get_modifier_name(modifier):
        """Get reproducible modifier names based on the modifier type."""
        if hasattr(modifier, 'name'):  # This is Pipeline or lookup table or something similar
            modifier_name = modifier.name
        elif hasattr(modifier, '__self__'):  # This is a bound method of a component or other object
            owner = modifier.__self__
            owner_name = owner.name if hasattr(owner, 'name') else owner.__class__.__name__
            modifier_name = f'{owner_name}.{modifier.__name__}'
        elif hasattr(modifier, '__name__'):  # Some unbound function
            modifier_name = modifier.__name__
        elif hasattr(modifier, '__call__'):  # Some anonymous callable
            modifier_name = f'{modifier.__class__.name__}.__call__'
        else:  # I don't know what this is.
            raise ValueError(f'Unknown modifier type: {type(modifier)}')
        return modifier_name

    def keys(self) -> Iterable[str]:
        """Get an iterable of pipeline names."""
        return self._pipelines.keys()

    def items(self) -> Iterable[Tuple[str, Pipeline]]:
        """Get an iterable of name, pipeline tuples."""
        return self._pipelines.items()

    def values(self) -> Iterable[Pipeline]:
        """Get an iterable of all pipelines."""
        return self._pipelines.values()

    def __contains__(self, item: str) -> bool:
        return item in self._pipelines

    def __iter__(self) -> Iterable[str]:
        return iter(self._pipelines)

    def __repr__(self) -> str:
        return "ValuesManager()"


class ValuesInterface:
    """Public interface for the simulation values management system.

    The values system provides tools to build up a value across many
    components, allowing users to build components that focus on small groups
    of simulant attributes.

    """

    def __init__(self, manager: ValuesManager):
        self._manager = manager

    def register_value_producer(self,
                                value_name: str,
                                source: Callable,
                                requires_columns: List[str] = (),
                                requires_values: List[str] = (),
                                requires_streams: List[str] = (),
                                preferred_combiner: Callable = replace_combiner,
                                preferred_post_processor: Callable = None) -> Callable:
        """Marks a ``Callable`` as the producer of a named value.

        Parameters
        ----------
        value_name
            The name of the new dynamic value pipeline.
        source
            A callable source for the dynamic value pipeline.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the pipeline source
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the pipeline source is called.
        requires_streams
            A list of the randomness streams that need to be properly sourced
            before the pipeline source is called.
        preferred_combiner
            A strategy for combining the source and the results of any calls
            to mutators in the pipeline. ``vivarium`` provides the strategies
            ``replace_combiner`` (the default) and ``list_combiner``, which
            are importable from ``vivarium.framework.values``.  Client code
            may define additional strategies as necessary.
        preferred_post_processor
            A strategy for processing the final output of the pipeline.
            ``vivarium`` provides the strategies ``rescale_post_processor``
            and ``union_post_processor`` which are importable from
            ``vivarium.framework.values``.  Client code may define additional
            strategies as necessary.

        Returns
        -------
            A callable reference to the named dynamic value pipeline.

        """
        return self._manager.register_value_producer(value_name, source,
                                                     requires_columns, requires_values, requires_streams,
                                                     preferred_combiner, preferred_post_processor)

    def register_rate_producer(self,
                               rate_name: str,
                               source: Callable[..., pd.DataFrame],
                               requires_columns: List[str] = (),
                               requires_values: List[str] = (),
                               requires_streams: List[str] = ()) -> Callable:
        """Marks a ``Callable`` as the producer of a named rate.

        This is a convenience wrapper around ``register_value_producer`` that
        makes sure rate data is appropriately scaled to the size of the
        simulation time step.  It is equivalent to
        ``register_value_producer(value_name, source,
        preferred_combiner=replace_combiner,
        preferred_post_processor=rescale_post_processor)``

        Parameters
        ----------
        rate_name
            The name of the new dynamic rate pipeline.
        source
            A callable source for the dynamic rate pipeline.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the pipeline source
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the pipeline source is called.
        requires_streams
            A list of the randomness streams that need to be properly sourced
            before the pipeline source is called.

        Returns
        -------
            A callable reference to the named dynamic rate pipeline.

        """
        return self.register_value_producer(rate_name, source,
                                            requires_columns, requires_values, requires_streams,
                                            preferred_post_processor=rescale_post_processor)

    def register_value_modifier(self,
                                value_name: str,
                                modifier: Callable,
                                requires_columns: List[str] = (),
                                requires_values: List[str] = (),
                                requires_streams: List[str] = (),):
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier should accept the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        requires_columns
            A list of the state table columns that already need to be present
            and populated in the state table before the pipeline modifier
            is called.
        requires_values
            A list of the value pipelines that need to be properly sourced
            before the pipeline modifier is called.
        requires_streams
            A list of the randomness streams that need to be properly sourced
            before the pipeline modifier is called.

        """
        self._manager.register_value_modifier(value_name, modifier,
                                              requires_columns, requires_values, requires_streams)

    def get_value(self, name: str) -> Pipeline:
        """Retrieve the pipeline representing the named value.

        Parameters
        ----------
        name
            Name of the pipeline to return.

        Returns
        -------
            A callable reference to the named pipeline.  The pipeline arguments
            should be identical to the arguments to the pipeline source
            (frequently just a :class:`pandas.Index` representing the
            simulants).

        """
        return self._manager.get_value(name)
