from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource import Resource
from vivarium.framework.values.combiners import ValueCombiner, replace_combiner
from vivarium.framework.values.pipeline import AttributePipeline, DynamicValueError, Pipeline
from vivarium.framework.values.post_processors import (
    AttributePostProcessor,
    PostProcessor,
    rescale_post_processor,
)
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    import pandas as pd

    from vivarium import Component
    from vivarium.framework.engine import Builder

T = TypeVar("T")


class ValuesManager(Manager):
    """Manager for the dynamic value system."""

    def __init__(self) -> None:
        # Pipelines are lazily initialized by _register_value_producer
        self._value_pipelines: dict[str, Pipeline] = {}
        self._attribute_pipelines: dict[str, AttributePipeline] = {}

    @property
    def name(self) -> str:
        return "values_manager"

    @property
    def _all_pipelines(self) -> dict[str, Pipeline]:
        return {**self._value_pipelines, **self._attribute_pipelines}

    def setup(self, builder: Builder) -> None:
        self.logger = builder.logging.get_logger(self.name)
        self.step_size = builder.time.step_size()
        self.simulant_step_sizes = builder.time.simulant_step_sizes()
        builder.event.register_listener("post_setup", self.on_post_setup)

        self.resources = builder.resources
        self.add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(
            self.register_value_producer, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_value_modifier, allow_during=[lifecycle_states.SETUP]
        )

    def on_post_setup(self, _event: Event) -> None:
        """Finalizes dependency structure for the pipelines."""
        # Unsourced pipelines might occur when generic components register
        # modifiers to values that aren't required in a simulation.
        unsourced_pipelines = [p for p, v in self.items() if not v.source]
        if unsourced_pipelines:
            self.logger.warning(f"Unsourced pipelines: {unsourced_pipelines}")

        # register_value_producer and register_value_modifier record the
        # dependency structure for the pipeline source and pipeline modifiers,
        # respectively. We don't have enough information to record the
        # dependency structure for the pipeline itself until now, where
        # we say the pipeline value depends on its source and all its
        # modifiers.
        for pipeline in self.values():
            self.resources.add_resources(
                pipeline.component, [pipeline], [pipeline.source] + list(pipeline.mutators)
            )

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        # TODO [MIC-5452]: all calls should have a component
        component: Component | None = None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | None = None,
    ) -> Pipeline:
        """Marks a ``Callable`` as the producer of a named value.

        See Also
        --------
            :meth:`ValuesInterface.register_value_producer`
        """
        self.logger.debug(f"Registering value pipeline {value_name}")
        pipeline = self.get_value(value_name)
        self._configure_pipeline(
            pipeline,
            source,
            component,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )
        return pipeline

    def register_attribute_producer(
        self,
        value_name: str,
        source: Callable[[pd.Index[int]], Any],
        component: Component | Manager,
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: AttributePostProcessor | None = None,
    ) -> AttributePipeline:
        """Marks a ``Callable`` as the producer of a named attribute.

        See Also
        --------
            :meth:`ValuesInterface.register_attribute_producer`
        """
        self.logger.debug(f"Registering attribute pipeline {value_name}")
        pipeline = self.get_attribute(value_name)
        self._configure_pipeline(
            pipeline,
            source,
            component,
            required_resources=required_resources,
            preferred_combiner=preferred_combiner,
            preferred_post_processor=preferred_post_processor,
        )
        return pipeline

    def register_value_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        # TODO [MIC-5452]: all calls should have a component
        component: Component | Manager | None = None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier must have the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        component
            The component that is registering the value modifier.
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
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._configure_modifier(
            self.get_value(value_name),
            modifier,
            component,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
        )

    def register_attribute_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        component: Component | Manager,
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named attribute.

        Parameters
        ----------
        value_name :
            The name of the dynamic attribute pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic attribute pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier must have the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        component
            The component that is registering the attribute modifier.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._configure_modifier(
            self.get_attribute(value_name),
            modifier,
            component,
            required_resources=required_resources,
        )

    def get_value(self, name: str) -> Pipeline:
        """Retrieve the pipeline representing the named value.

        Parameters
        ----------
        name
            Name of the pipeline to return.

        Returns
        -------
            A callable reference to the named pipeline. The pipeline arguments
            should be identical to the arguments to the pipeline source
            (frequently just a :class:`pandas.Index` representing the
            simulants).
        """
        if name in self._attribute_pipelines:
            raise DynamicValueError(
                f"'{name}' is already registered as an attribute pipeline."
            )
        pipeline = self._value_pipelines.get(name) or Pipeline(name)
        self._value_pipelines[name] = pipeline
        return pipeline

    def get_attribute(self, name: str) -> AttributePipeline:
        """Retrieve the pipeline representing the named attribute.

        Parameters
        ----------
        name
            Name of the attribute pipeline to return.

        Returns
        -------
            A callable reference to the named attribute pipeline. The single
            attribute pipeline argument must a :class:`pandas.Index` representing
            the simulants and must return a :class:`pandas.DataFrame` with that same index.
        """
        if name in self._value_pipelines:
            raise DynamicValueError(f"'{name}' is already registered as a value pipeline.")
        pipeline = self._attribute_pipelines.get(name) or AttributePipeline(name)
        self._attribute_pipelines[name] = pipeline
        return pipeline

    ##################
    # Helper methods #
    ##################

    def _configure_pipeline(
        self,
        pipeline: Pipeline | AttributePipeline,
        source: Callable[..., Any],
        component: Component | Manager | None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | AttributePostProcessor | None = None,
    ) -> None:
        pipeline.set_attributes(
            component,
            source,
            preferred_combiner,
            preferred_post_processor,
            self,
        )

        # The resource we add here is just the pipeline source.
        # The value will depend on the source and its modifiers, and we'll
        # declare that resource at post-setup once all sources and modifiers
        # are registered.
        dependencies = self._convert_dependencies(
            source, requires_columns, requires_values, requires_streams, required_resources
        )
        self.resources.add_resources(pipeline.component, [pipeline.source], dependencies)

        self.add_constraint(
            pipeline._call,
            restrict_during=[
                lifecycle_states.INITIALIZATION,
                lifecycle_states.SETUP,
                lifecycle_states.POST_SETUP,
            ],
        )

    def _configure_modifier(
        self,
        pipeline: Pipeline | AttributePipeline,
        modifier: Callable[..., Any],
        component: Component | Manager | None = None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        value_modifier = pipeline.get_value_modifier(modifier, component)
        self.logger.debug(f"Registering {value_modifier.name} as modifier to {pipeline.name}")
        dependencies = self._convert_dependencies(
            modifier, requires_columns, requires_values, requires_streams, required_resources
        )
        self.resources.add_resources(component, [value_modifier], dependencies)

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
                + [Resource("value", name, None) for name in requires_values]
                + [Resource("stream", name, None) for name in requires_streams]
            )
        else:
            return required_resources

    def keys(self) -> Iterable[str]:
        """Get an iterable of pipeline names."""
        return self._all_pipelines.keys()

    def items(self) -> Iterable[tuple[str, Pipeline]]:
        """Get an iterable of name, pipeline tuples."""
        return self._all_pipelines.items()

    def values(self) -> Iterable[Pipeline]:
        """Get an iterable of all pipelines."""
        return self._all_pipelines.values()

    def __contains__(self, item: str) -> bool:
        return item in self._all_pipelines

    def __iter__(self) -> Iterable[str]:
        return iter(self._all_pipelines)

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
        # TODO [MIC-5452]: all calls should have a component
        component: Component | None = None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
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
        component
            The component that is registering the value producer.
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
            pipeline source is called. This is a list of strings, pipelines,
            or randomness streams.
        preferred_combiner
            A strategy for combining the source and the results of any calls
            to mutators in the pipeline. ``vivarium`` provides the strategies
            ``replace_combiner`` (the default) and ``list_combiner``, which
            are importable from ``vivarium.framework.values``. Client code
            may define additional strategies as necessary.
        preferred_post_processor
            A strategy for processing the final output of the pipeline.
            ``vivarium`` provides the strategies ``rescale_post_processor``
            and ``union_post_processor`` which are importable from
            ``vivarium.framework.values``. Client code may define additional
            strategies as necessary.

        Returns
        -------
            A callable reference to the named dynamic value pipeline.
        """
        return self._manager.register_value_producer(
            value_name,
            source,
            component,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )

    def register_attribute_producer(
        self,
        value_name: str,
        source: Callable[[pd.Index[int]], Any],
        component: Component | Manager,
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: AttributePostProcessor | None = None,
    ) -> AttributePipeline:
        """Marks a ``Callable`` as the producer of a named attribute.

        Parameters
        ----------
        value_name
            The name of the new dynamic attribute pipeline.
        source
            A callable source for the dynamic attribute pipeline.
        component
            The component that is registering the attribute producer.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called. This is a list of strings, pipelines,
            or randomness streams.
        preferred_combiner
            A strategy for combining the source and the results of any calls
            to mutators in the pipeline. ``vivarium`` provides the strategies
            ``replace_combiner`` (the default) and ``list_combiner``, which
            are importable from ``vivarium.framework.values``. Client code
            may define additional strategies as necessary.
        preferred_post_processor
            A strategy for processing the final output of the pipeline.
            ``vivarium`` provides the strategies ``rescale_post_processor``
            and ``union_post_processor`` which are importable from
            ``vivarium.framework.values``. Client code may define additional
            strategies as necessary.

        Returns
        -------
            A callable reference to the named dynamic attribute pipeline.
        """
        return self._manager.register_attribute_producer(
            value_name,
            source,
            component,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )

    def register_rate_producer(
        self,
        rate_name: str,
        source: Callable[..., Any],
        component: Component,
        required_resources: Sequence[str | Resource] = (),
    ) -> AttributePipeline:
        """Marks a ``Callable`` as the producer of a named rate.

        This is a convenience wrapper around ``register_attribute_producer`` that
        makes sure rate data is appropriately scaled to the size of the
        simulation time step. It is equivalent to
        ``register_attribute_producer(value_name, source,
        preferred_combiner=replace_combiner,
        preferred_post_processor=rescale_post_processor)``

        Parameters
        ----------
        rate_name
            The name of the new dynamic rate pipeline.
        source
            A callable source for the dynamic rate pipeline.
        component
            The component that is registering the rate producer.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called. This is a list of strings, pipelines,
            or randomness streams.

        Returns
        -------
            A callable reference to the named dynamic rate pipeline.
        """
        return self.register_attribute_producer(
            rate_name,
            source,
            component,
            required_resources,
            preferred_post_processor=rescale_post_processor,
        )

    def register_value_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        # TODO [MIC-5452]: all calls should have a component
        component: Component | Manager | None = None,
        requires_columns: Iterable[str] = (),
        requires_values: Iterable[str] = (),
        requires_streams: Iterable[str] = (),
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name :
            The name of the dynamic value pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic value pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier must have the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        component
            The component that is registering the value modifier.
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
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._manager.register_value_modifier(
            value_name,
            modifier,
            component,
            requires_columns,
            requires_values,
            requires_streams,
            required_resources,
        )

    def register_attribute_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
        component: Component | Manager,
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named attribute.

        Parameters
        ----------
        value_name :
            The name of the dynamic attribute pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic attribute pipeline
            when called. If the pipeline has a ``replace_combiner``, the
            modifier should accept the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        component
            The component that is registering the attribute modifier.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._manager.register_attribute_modifier(
            value_name,
            modifier,
            component,
            required_resources=required_resources,
        )

    def get_value(self, name: str) -> Pipeline:
        """Retrieve the pipeline representing the named value.

        Parameters
        ----------
        name
            Name of the pipeline to return.

        Returns
        -------
            A callable reference to the named pipeline. The pipeline arguments
            should be identical to the arguments to the pipeline source
            (frequently just a :class:`pandas.Index` representing the
            simulants).

        """
        return self._manager.get_value(name)

    # TODO: [MIC-5452] Remove this method (attributes should be obtained via population views)
    def get_attribute(self, name: str) -> AttributePipeline:
        """A temporary interface method to use while during population re-design."""
        return self._manager.get_attribute(name)
