"""
=====================
Values System Manager
=====================

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource import Resource
from vivarium.framework.values.combiners import ValueCombiner, replace_combiner
from vivarium.framework.values.pipeline import (
    AttributePipeline,
    AttributesValueSource,
    DynamicValueError,
    Pipeline,
    PrivateColumnValueSource,
    ValueSource,
)
from vivarium.framework.values.post_processors import AttributePostProcessor, PostProcessor
from vivarium.manager import Manager

if TYPE_CHECKING:
    import pandas as pd

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
        self._population_mgr = builder.population._manager
        self.logger = builder.logging.get_logger(self.name)
        self.step_size = builder.time.step_size()
        self.simulant_step_sizes = builder.time.simulant_step_sizes()
        builder.event.register_listener("post_setup", self.on_post_setup)

        self._add_resources = builder.resources.add_resources
        self._get_current_component = builder.components.get_current_component_or_manager
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(
            self.register_value_producer, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_value_modifier, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.get_attribute_pipelines, restrict_during=[lifecycle_states.SETUP]
        )

    def on_post_setup(self, _event: Event) -> None:
        """Finalizes dependency structure for the pipelines."""
        for pipeline in self._all_pipelines.values():
            # Unsourced pipelines might occur when generic components register
            # modifiers to values that aren't required in a simulation.
            if not pipeline.source:
                self.logger.warning(
                    f"Pipeline {pipeline.name} has no source. It will not be usable."
                )
                continue

            # register_value_producer and register_value_modifier record the
            # dependency structure for the pipeline source and pipeline modifiers,
            # respectively. We don't have enough information to record the
            # dependency structure for the pipeline itself until now, where
            # we say the pipeline value depends on its source and all its
            # modifiers.
            self._add_resources(
                component=pipeline.component,
                resources=pipeline,
                dependencies=[pipeline.source] + list(pipeline.mutators),
            )

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
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
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )
        return pipeline

    def register_attribute_producer(
        self,
        value_name: str,
        source: Callable[[pd.Index[int]], Any] | list[str],
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: AttributePostProcessor | None = None,
        source_is_private_column: bool = False,
    ) -> None:
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
            required_resources=required_resources,
            preferred_combiner=preferred_combiner,
            preferred_post_processor=preferred_post_processor,
            source_is_private_column=source_is_private_column,
        )

    def register_value_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any],
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
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._configure_modifier(
            self.get_value(value_name),
            modifier,
            required_resources,
        )

    def register_attribute_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any] | str,
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named attribute.

        Parameters
        ----------
        value_name :
            The name of the dynamic attribute pipeline to be modified.
        modifier :
            A function that modifies the source of the dynamic attribute pipeline
            when called; if a string is passed, it refers to the name of an
            :class:`~vivarium.framework.values.pipeline.AttributePipeline`.
            If the pipeline has a ``replace_combiner``, the
            modifier should accept the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        modifier = self.get_attribute(modifier) if isinstance(modifier, str) else modifier
        self._configure_modifier(
            self.get_attribute(value_name),
            modifier,
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
        pipeline = self._value_pipelines.get(name, Pipeline(name))
        self._value_pipelines[name] = pipeline
        return pipeline

    def get_value_pipelines(self) -> dict[str, Pipeline]:
        return self._value_pipelines

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
        pipeline = self._attribute_pipelines.get(name, AttributePipeline(name))
        self._attribute_pipelines[name] = pipeline
        return pipeline

    def get_attribute_pipelines(self) -> dict[str, AttributePipeline]:
        return self._attribute_pipelines

    ##################
    # Helper methods #
    ##################

    def _configure_pipeline(
        self,
        pipeline: Pipeline | AttributePipeline,
        source: Callable[..., Any] | list[str],
        required_resources: Sequence[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | AttributePostProcessor | None = None,
        source_is_private_column: bool = False,
    ) -> None:
        component = self._get_current_component()
        value_source: ValueSource
        if source_is_private_column:
            value_source = PrivateColumnValueSource(
                pipeline, source, component, required_resources
            )
        elif isinstance(source, list):
            value_source = AttributesValueSource(
                pipeline, source, component, required_resources
            )
        else:
            value_source = ValueSource(pipeline, source, component, required_resources)

        pipeline.set_attributes(
            component=component,
            source=value_source,
            combiner=preferred_combiner,
            post_processor=preferred_post_processor,  # type: ignore[arg-type]
            manager=self,
        )

        # The resource we add here is just the pipeline source.
        self._add_resources(
            component=pipeline.component,
            resources=pipeline.source,
            dependencies=pipeline.source.required_resources,
        )

        self._add_constraint(
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
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        component = self._get_current_component()
        value_modifier = pipeline.get_value_modifier(modifier, component)
        self.logger.debug(f"Registering {value_modifier.name} as modifier to {pipeline.name}")
        if isinstance(modifier, Resource) and required_resources:
            self.logger.warning(
                f"Conflicting information for {pipeline.name}. Ignoring 'required_resources' "
                f"since the `modifier` is of type {type(modifier)} and we can infer "
                "the required resources directly."
            )
            required_resources = [modifier]
        self._add_resources(
            component=component, resources=value_modifier, dependencies=required_resources
        )

    def __contains__(self, item: str) -> bool:
        return item in self._all_pipelines

    def __iter__(self) -> Iterable[str]:
        return iter(self._all_pipelines)

    def __repr__(self) -> str:
        return "ValuesManager()"
