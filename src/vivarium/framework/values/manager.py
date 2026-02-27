"""
==============
Values Manager
==============

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeVar

from vivarium.framework.event import Event
from vivarium.framework.lifecycle import lifecycle_states
from vivarium.framework.resource import Column, Resource
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
    """Manager for the dynamic value system.

    Notes
    -----
    This is the only manager for the values system; different methods exist for
    working with generic value :class:`Pipelines <vivarium.framework.values.pipeline.Pipeline>`
    and :class:`AttributePipelines <vivarium.framework.values.pipeline.AttributePipeline>`.
    """

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

        self._get_view = builder.population.get_view
        self._add_resource = builder.resources.add_resource
        self._get_current_component = builder.components.get_current_component_or_manager
        self._add_constraint = builder.lifecycle.add_constraint

        builder.lifecycle.add_constraint(
            self.register_value_producer, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_value_modifier, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_attribute_producer, allow_during=[lifecycle_states.SETUP]
        )
        builder.lifecycle.add_constraint(
            self.register_attribute_modifier, allow_during=[lifecycle_states.SETUP]
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

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        required_resources: Iterable[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | None = None,
    ) -> Pipeline:
        """Registers a ``Pipeline`` as the producer of a named value.

        Parameters
        ----------
        value_name
            The name of the new dynamic value pipeline.
        source
            A callable source for the dynamic value pipeline.
        required_resources
            A list of resources that the producer requires. A string represents
            a population attribute.
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
            The ``Pipeline`` that is registered as the producer of the named value.
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
        required_resources: Iterable[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: AttributePostProcessor | None = None,
        source_is_private_column: bool = False,
    ) -> None:
        """Registers an ``AttributePipeline`` as the producer of a named attribute.

        Parameters
        ----------
        value_name
            The name of the new dynamic attribute pipeline.
        source
            The source for the dynamic attribute pipeline. This can be a callable,
            a list containing a single name of a private column created by this
            component, or a list of population attributes. If a private column name
            is passed, `source_is_private_column` must also be set to True.
        required_resources
            A list of resources that the producer requires. A string represents
            a population attribute.
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
        source_is_private_column
            Whether or not the source is the name of a private column created by
            this component.
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
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named value.

        Parameters
        ----------
        value_name
            The name of the dynamic value ``Pipeline`` to be modified.
        modifier
            A function that modifies the source of the dynamic value ``Pipeline``
            when called. If the pipeline has a ``replace_combiner``, the
            modifier must have the same arguments as the pipeline source
            with an additional last positional argument for the results of the
            previous stage in the pipeline. For the ``list_combiner`` strategy,
            the pipeline modifiers should have the same signature as the pipeline
            source.
        required_resources
            A list of resources that the producer requires. A string represents
            a population attribute.
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
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        """Marks a ``Callable`` as the modifier of a named attribute.

        Parameters
        ----------
        value_name
            The name of the dynamic ``AttributePipeline`` to be modified.
        modifier
            A function that modifies the source of the dynamic ``AttributePipeline``
            when called. If a string is passed, it refers to the name of an ``AttributePipeline``.
            If the pipeline has a ``replace_combiner``, the modifier should accept
            the same arguments as the pipeline source with an additional last positional
            argument for the results of the previous stage in the pipeline. For
            the ``list_combiner`` strategy, the pipeline modifiers should have the
            same signature as the pipeline source.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of attribute names, pipelines,
            or randomness streams.
        """
        modifier = self.get_attribute(modifier) if isinstance(modifier, str) else modifier
        self._configure_modifier(
            self.get_attribute(value_name),
            modifier,
            required_resources=required_resources,
        )

    def get_value(self, name: str) -> Pipeline:
        """Retrieves the ``Pipeline`` representing the named value.

        Parameters
        ----------
        name
            Name of the ``Pipeline`` to return.

        Returns
        -------
            The requested ``Pipeline``.

        Notes
        -----
        This will create a new ``Pipeline`` if one does not already exist.
        """
        if name in self._attribute_pipelines:
            raise DynamicValueError(
                f"'{name}' is already registered as an attribute pipeline."
            )
        pipeline = self._value_pipelines.get(name, Pipeline(name))
        self._value_pipelines[name] = pipeline
        return pipeline

    def get_value_pipelines(self) -> dict[str, Pipeline]:
        """Retrieves a dictionary of all registered value ``Pipelines``.

        To get all ``AttributePipelines``, use :meth:`get_attribute_pipelines`.

        Returns
        -------
            A dictionary mapping value names to their corresponding ``Pipelines``.
        """
        return self._value_pipelines

    def get_attribute(self, name: str) -> AttributePipeline:
        """Retrieves the ``AttributePipeline`` representing the named attribute.

        To get a value ``Pipeline``, use :meth:`get_value`.

        Parameters
        ----------
        name
            Name of the ``AttributePipeline`` to return.

        Returns
        -------
            The requested ``AttributePipeline``.

        Notes
        -----
        This will create a new ``AttributePipeline`` if one does not already exist.
        """
        if name in self._value_pipelines:
            raise DynamicValueError(f"'{name}' is already registered as a value pipeline.")
        pipeline = self._attribute_pipelines.get(name, AttributePipeline(name))
        self._attribute_pipelines[name] = pipeline
        return pipeline

    def get_attribute_pipelines(self) -> dict[str, AttributePipeline]:
        """Returns a dictionary of ``AttributePipelines``.

        Returns
        -------
            A dictionary mapping all registered attribute names to their corresponding
            ``AttributePipelines``.

        Notes
        -----
        This is not the preferred access method to getting population attributes
        since it does not implement various features (e.g. querying, simulant
        tracking, etc); it exists for other managers to use if needed. Use
        :meth:`vivarium.framework.population.population_view.PopulationView.get_attributes`
        or :meth:`vivarium.framework.population.population_view.PopulationView.get_attribute_frame`
        instead.
        """
        return self._attribute_pipelines

    ##################
    # Helper methods #
    ##################

    def _configure_pipeline(
        self,
        pipeline: Pipeline | AttributePipeline,
        source: Callable[..., Any] | list[str],
        required_resources: Iterable[str | Resource] = (),
        preferred_combiner: ValueCombiner = replace_combiner,
        preferred_post_processor: PostProcessor | AttributePostProcessor | None = None,
        source_is_private_column: bool = False,
    ) -> None:
        component = self._get_current_component()
        value_source: ValueSource
        if source_is_private_column:
            generic_error_msg = (
                f"Invalid source for {pipeline.name}. `source` must be list containing a single"
                " private column name."
            )
            if not isinstance(source, list):
                raise ValueError(
                    generic_error_msg + f"Got `source` type {type(source)} instead."
                )
            if len(source) != 1:
                raise ValueError(generic_error_msg + f"Got {len(source)} names instead.")
            value_source = PrivateColumnValueSource(
                pipeline, source[0], self._get_view(component)  # type: ignore[arg-type]
            )
            if required_resources:
                self.logger.warning(
                    f"Conflicting information for {pipeline.name}. Ignoring 'required_resources' "
                    "since the `source_is_private_column` flag is set to True and we can infer "
                    "the required resources directly."
                )
            required_resources = [Column(source[0], component)]
        elif isinstance(source, list):
            value_source = AttributesValueSource(pipeline, source, self._get_view(component))  # type: ignore[arg-type]
            if required_resources:
                self.logger.warning(
                    f"Conflicting information for {pipeline.name}. Ignoring 'required_resources' "
                    "since the `source` is a list of attributes and we can infer the required "
                    "resources directly."
                )
            required_resources = source
        else:
            value_source = ValueSource(pipeline, source)

        pipeline.set_attributes(
            component=component,
            source=value_source,
            combiner=preferred_combiner,
            post_processor=preferred_post_processor,  # type: ignore[arg-type]
            required_resources=required_resources,
            manager=self,
        )

        self._add_resource(pipeline)
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
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        component = self._get_current_component()
        if isinstance(modifier, Resource):
            if required_resources:
                self.logger.warning(
                    f"Conflicting information for {pipeline.name}. Ignoring 'required_resources' "
                    f"since the `modifier` is of type {type(modifier)} and we can infer "
                    "the required resources directly."
                )
            required_resources = [modifier]
        value_modifier = pipeline.get_value_modifier(modifier, component, required_resources)
        self.logger.debug(f"Registering {value_modifier.name} as modifier to {pipeline.name}")
        self._add_resource(value_modifier)

    def __contains__(self, item: str) -> bool:
        return item in self._all_pipelines

    def __iter__(self) -> Iterable[str]:
        return iter(self._all_pipelines)

    def __repr__(self) -> str:
        return "ValuesManager()"
