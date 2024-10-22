from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from vivarium.framework.event import Event
from vivarium.framework.randomness import RandomnessStream
from vivarium.framework.resource import Resource
from vivarium.framework.values.combiners import ValueCombiner, replace_combiner
from vivarium.framework.values.pipeline import (
    MissingValueSource,
    Pipeline,
    ValueModifier,
    ValueSource,
)
from vivarium.framework.values.post_processors import PostProcessor, rescale_post_processor
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder

T = TypeVar("T")


class ValuesManager(Manager):
    """Manager for the dynamic value system."""

    def __init__(self) -> None:
        # Pipelines are lazily initialized by _register_value_producer
        self._pipelines: dict[str, Pipeline] = {}

    @property
    def name(self) -> str:
        return "values_manager"

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)
        self.step_size = builder.time.step_size()
        self.simulant_step_sizes = builder.time.simulant_step_sizes()
        builder.event.register_listener("post_setup", self.on_post_setup)

        self.resources = builder.resources
        self.add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(self.register_value_producer, allow_during=["setup"])
        builder.lifecycle.add_constraint(self.register_value_modifier, allow_during=["setup"])

    def on_post_setup(self, _event: Event) -> None:
        """Finalizes dependency structure for the pipelines."""
        # Unsourced pipelines might occur when generic components register
        # modifiers to values that aren't required in a simulation.
        unsourced_pipelines = [p for p, v in self._pipelines.items() if v.source is None]
        if unsourced_pipelines:
            self.logger.warning(f"Unsourced pipelines: {unsourced_pipelines}")

        # register_value_producer and register_value_modifier record the
        # dependency structure for the pipeline source and pipeline modifiers,
        # respectively.  We don't have enough information to record the
        # dependency structure for the pipeline itself until now, where
        # we say the pipeline value depends on its source and all its
        # modifiers.
        for name, pipe in self._pipelines.items():
            dependencies: list[Resource] = [
                ValueSource(name) if pipe.source else MissingValueSource(name)
            ]
            for i, m in enumerate(pipe.mutators):
                mutator_name = self._get_modifier_name(m)
                dependencies.append(ValueModifier(f"{name}.{i + 1}.{mutator_name}"))
            self.resources.add_resources([pipe], dependencies)

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Pipeline | RandomnessStream] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | None = None,
    ) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named value.

        See Also
        --------
            :meth:`ValuesInterface.register_value_producer`
        """
        pipeline = self._register_value_producer(
            value_name, source, preferred_combiner, preferred_post_processor
        )

        # The resource we add here is just the pipeline source.
        # The value will depend on the source and its modifiers, and we'll
        # declare that resource at post-setup once all sources and modifiers
        # are registered.
        dependencies = self._convert_dependencies(
            source, requires_columns, requires_values, requires_streams, required_resources
        )
        self.resources.add_resources([ValueSource(value_name)], dependencies)
        self.add_constraint(
            pipeline._call, restrict_during=["initialization", "setup", "post_setup"]
        )

        return pipeline

    def _register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        preferred_combiner: ValueCombiner,
        preferred_post_processor: PostProcessor | None,
    ) -> Pipeline:
        """Configure the named value pipeline with a source, combiner, and post-processor."""
        self.logger.debug(f"Registering value pipeline {value_name}")
        pipeline = self.get_value(value_name)
        Pipeline.setup_pipeline(
            pipeline, source, preferred_combiner, preferred_post_processor, self
        )
        return pipeline

    def register_value_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Pipeline | RandomnessStream] = (),
    ) -> None:
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
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called.  This is a list of strings, pipeline
            names, or randomness streams.
        """
        modifier_name = self._get_modifier_name(modifier)

        pipeline = self.get_value(value_name)
        pipeline.mutators.append(modifier)

        name = f"{value_name}.{len(pipeline.mutators)}.{modifier_name}"
        self.logger.debug(f"Registering {name} as modifier to {value_name}")
        dependencies = self._convert_dependencies(
            modifier, requires_columns, requires_values, requires_streams, required_resources
        )
        self.resources.add_resources([ValueModifier(name)], dependencies)

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
        pipeline = self._pipelines.get(name) or Pipeline(name)
        self._pipelines[name] = pipeline
        return pipeline

    @staticmethod
    def _convert_dependencies(
        func: Callable[..., Any],
        requires_columns: Iterable[str],
        requires_values: Iterable[str],
        requires_streams: Iterable[str],
        required_resources: Iterable[str | Resource],
    ) -> Iterable[str | Resource]:
        if isinstance(func, Pipeline):
            # The dependencies of the pipeline itself will have been declared
            # when the pipeline was registered.
            return [func]

        if requires_columns or requires_values or requires_streams:
            warnings.warn(
                "Specifying requirements individually is deprecated. You should "
                "specify them using the 'required_resources' argument instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if required_resources:
                raise ValueError(
                    "If requires_columns, requires_values, or requires_streams"
                    " are provided, requirements must be empty."
                )

            return (
                list(requires_columns)
                + [Resource("value", name) for name in requires_values]
                + [Resource("stream", name) for name in requires_streams]
            )
        else:
            return required_resources

    @staticmethod
    def _get_modifier_name(modifier: Callable[..., Any]) -> str:
        """Get reproducible modifier names based on the modifier type."""
        if hasattr(modifier, "name"):  # This is Pipeline or lookup table or something similar
            modifier_name: str = modifier.name
        elif hasattr(modifier, "__self__") and hasattr(
            modifier, "__name__"
        ):  # This is a bound method of a component or other object
            owner = modifier.__self__
            owner_name = owner.name if hasattr(owner, "name") else owner.__class__.__name__
            modifier_name = f"{owner_name}.{modifier.__name__}"
        elif hasattr(modifier, "__name__"):  # Some unbound function
            modifier_name = modifier.__name__
        elif hasattr(modifier, "__call__"):  # Some anonymous callable
            modifier_name = f"{modifier.__class__.__name__}.__call__"
        else:  # I don't know what this is.
            raise ValueError(f"Unknown modifier type: {type(modifier)}")
        return modifier_name

    def keys(self) -> Iterable[str]:
        """Get an iterable of pipeline names."""
        return self._pipelines.keys()

    def items(self) -> Iterable[tuple[str, Pipeline]]:
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


class ValuesInterface(Interface):
    """Public interface for the simulation values management system.

    The values system provides tools to build up a value across many
    components, allowing users to build components that focus on small groups
    of simulant attributes.

    """

    def __init__(self, manager: ValuesManager) -> None:
        self._manager = manager

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Pipeline | RandomnessStream] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | None = None,
    ) -> Pipeline:
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
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called.  This is a list of strings, pipeline
            names, or randomness streams.
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
        return self._manager.register_value_producer(
            value_name,
            source,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )

    def register_rate_producer(
        self,
        rate_name: str,
        source: Callable[..., Any],
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Pipeline | RandomnessStream] = (),
    ) -> Pipeline:
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
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called.  This is a list of strings, pipeline
            names, or randomness streams.

        Returns
        -------
            A callable reference to the named dynamic rate pipeline.
        """
        return self.register_value_producer(
            rate_name,
            source,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
            preferred_post_processor=rescale_post_processor,
        )

    def register_value_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Pipeline | RandomnessStream] = (),
    ) -> None:
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
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called.  This is a list of strings, pipeline
            names, or randomness streams.
        """
        self._manager.register_value_modifier(
            value_name,
            modifier,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
        )

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
