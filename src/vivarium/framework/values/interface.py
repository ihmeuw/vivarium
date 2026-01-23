"""
================
Values Interface
================

This module provides a :class:`ValuesInterface <ValuesInterface>` class with
methods to register different types of value and attribute producers and modifiers.

"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from vivarium.framework.resource import Resource
from vivarium.framework.values.combiners import ValueCombiner, replace_combiner
from vivarium.framework.values.pipeline import AttributePipeline, Pipeline
from vivarium.framework.values.post_processors import (
    AttributePostProcessor,
    PostProcessor,
    rescale_post_processor,
)
from vivarium.manager import Interface

if TYPE_CHECKING:
    import pandas as pd

    from vivarium.framework.values import ValuesManager


class ValuesInterface(Interface):
    """Public interface for the simulation values management system.

    The values system provides tools to build up values across many components,
    allowing users to build components that focus on small groups of simulation
    variables.

    Notes
    -----
    This is the only public interface for the values system; different methods
    exist for working with generic value :class:`Pipelines <vivarium.framework.values.pipeline.Pipeline>`
    and :class:`AttributePipelines <vivarium.framework.values.pipeline.AttributePipeline>`.

    """

    def __init__(self, manager: ValuesManager) -> None:
        self._manager = manager

    def register_value_producer(
        self,
        value_name: str,
        source: Callable[..., Any],
        required_resources: Sequence[str | Resource] = (),
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
            The ``Pipeline`` that is registered as the producer of the named value.
        """
        return self._manager.register_value_producer(
            value_name,
            source,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
        )

    def register_attribute_producer(
        self,
        value_name: str,
        source: Callable[[pd.Index[int]], Any] | list[str],
        required_resources: Sequence[str | Resource] = (),
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
        source_is_private_column
            Whether or not the source is the name of a private column created by
            this component.
        """
        self._manager.register_attribute_producer(
            value_name,
            source,
            required_resources,
            preferred_combiner,
            preferred_post_processor,
            source_is_private_column,
        )

    def register_rate_producer(
        self,
        rate_name: str,
        source: Callable[[pd.Index[int]], Any] | list[str],
        required_resources: Sequence[str | Resource] = (),
    ) -> None:
        """Registers an ``AttributePipeline`` as the producer of a named rate.

        This is a convenience wrapper around ``register_attribute_producer`` that
        makes sure rate data is appropriately scaled to the size of the simulation
        time step. It is equivalent to calling ``register_attribute_producer()``
        with the ``rescale_post_processor`` as the preferred post processor.

        Parameters
        ----------
        rate_name
            The name of the new dynamic rate pipeline.
        source
            The source for the dynamic rate pipeline. This can be a callable
            or a list of column names. If a list of column names is provided,
            the component that is registering this attribute producer must be the
            one that creates those columns.
        required_resources
            A list of resources that need to be properly sourced before the
            pipeline source is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self.register_attribute_producer(
            rate_name,
            source,
            required_resources,
            preferred_post_processor=rescale_post_processor,
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
            A list of resources that need to be properly sourced before the
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._manager.register_value_modifier(value_name, modifier, required_resources)

    def register_attribute_modifier(
        self,
        value_name: str,
        modifier: Callable[..., Any] | str,
        required_resources: Sequence[str | Resource] = (),
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
            pipeline modifier is called. This is a list of strings, pipelines,
            or randomness streams.
        """
        self._manager.register_attribute_modifier(
            value_name,
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
        return self._manager.get_value(name)

    def get_attribute_pipelines(self) -> Callable[[], dict[str, AttributePipeline]]:
        """Returns a ``Callable`` that retrieves a dictionary of ``AttributePipelines``.

        Returns
        -------
            A ``Callable`` that returns a dictionary mapping all registered attribute
            names to their corresponding ``AttributePipelines``.

        Notes
        -----
        This is not the preferred access method to getting population attributes
        since it does not implement various features (e.g. querying, simulant
        tracking, etc); it exists for other managers to use if needed. Use
        :meth:`vivarium.framework.population.population_view.PopulationView.get_attributes`
        or :meth:`vivarium.framework.population.population_view.PopulationView.get_attribute_frame`
        instead.
        """
        return self._manager.get_attribute_pipelines
