"""
================
Values Interface
================

This module provides a :class:`ValuesInterface <ValuesInterface>` class with
methods to register different types of value and attribute producers and modifiers.

"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

from vivarium.framework.resource import Resource
from vivarium.framework.values.combiners import ValueCombiner, replace_combiner
from vivarium.framework.values.pipeline import AttributePipeline, Pipeline
from vivarium.framework.values.post_processors import (
    AttributePostProcessor,
    PostProcessor,
    rescale_post_processor,
)
from vivarium.manager import Interface, Manager

if TYPE_CHECKING:
    import pandas as pd

    from vivarium import Component
    from vivarium.framework.values import ValuesManager


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
        # TODO [MIC-6433]: all calls should have a component
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
        source: Callable[[pd.Index[int]], Any] | list[str],
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
            The source for the dynamic attribute pipeline. This can be a callable
            or a list of column names. If a list of column names is provided,
            they will be treated as private; the component that is registering
            this attribute producer must be the one that creates them and they
            will not be accessible by other components.
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
        source: Callable[[pd.Index[int]], Any] | list[str],
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
            The source for the dynamic rate pipeline. This can be a callable
            or a list of column names. If a list of column names is provided,
            the component that is registering this attribute producer must be the
            one that creates those columns.
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
        # TODO [MIC-6433]: all calls should have a component
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

    # TODO: [MIC-6388] Remove this method (attributes should be obtained via population views)
    def get_attribute(self, name: str) -> AttributePipeline:
        """A temporary interface method to use while during population re-design."""
        return self._manager.get_attribute(name)

    def get_attribute_pipelines(self) -> Callable[[], dict[str, AttributePipeline]]:
        return self._manager.get_attribute_pipelines
