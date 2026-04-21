from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any, Literal, TypeVar

import pandas as pd

from vivarium import Component
from vivarium.framework.resource import Resource
from vivarium.framework.values.exceptions import DynamicValueError
from vivarium.manager import Manager

if TYPE_CHECKING:
    from vivarium.framework.population import PopulationView
    from vivarium.framework.values import (
        AttributePostProcessor,
        PostProcessor,
        ValueCombiner,
        ValuesManager,
    )

T = TypeVar("T")


class ValueSource:
    """A wrapper for the source of a value pipeline."""

    def __init__(self, pipeline: Pipeline, source: Callable[..., Any]) -> None:
        self._pipeline = pipeline
        self._source = source

    def __bool__(self) -> bool:
        return True

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._source(*args, **kwargs)


class MissingValueSource(ValueSource):
    """A placeholder value source representing a pipeline with no source.

    This is used when a modifier of a pipeline is registered before the pipeline itself.
    The source of the pipeline must be set before the pipeline is called, but this allows
    for more flexible ordering of component setup.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def __bool__(self) -> bool:
        return False

    def _source(self, *args: Any, **kwargs: Any) -> Any:
        raise DynamicValueError(
            f"The pipeline for {self._pipeline.name} has no source. This likely means you are"
            " attempting to modify a value that hasn't been created."
        )


class PrivateColumnValueSource(ValueSource):
    """A value source representing a private column source of a value pipeline."""

    def __init__(
        self, pipeline: Pipeline, source: str, population_view: PopulationView
    ) -> None:
        self._pipeline = pipeline
        self.column_name = source
        """The name of the private column that is the source of this pipeline."""
        self._population_view = population_view
        """A population view that can be used to access the private column source of this pipeline."""

    def _source(self, index: pd.Index[int]) -> pd.Series[Any]:
        return self._population_view._manager.get_private_columns(
            component=self._pipeline.component, index=index, columns=self.column_name
        )


class AttributesValueSource(ValueSource):
    """A value source representing the list of attributes source of an attribute pipeline."""

    def __init__(
        self, pipeline: Pipeline, source: list[str], population_view: PopulationView
    ) -> None:
        self._pipeline = pipeline
        self.attributes = source[0] if len(source) == 1 else source
        """The name or list of names of the attributes that are the source of this pipeline."""
        self._population_view = population_view
        """A population view that can be used to access the attribute source of this pipeline."""

    def _source(self, index: pd.Index[int]) -> pd.Series[Any] | pd.DataFrame:
        return self._population_view.get(index=index, attributes=self.attributes)


class ValueModifier(Resource):
    """A resource representing a modifier of a value pipeline."""

    RESOURCE_TYPE = "value_modifier"

    def __init__(
        self,
        pipeline: Pipeline,
        modifier: Callable[..., Any],
        component: Component | Manager,
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        mutator_name = self.get_callable_name(modifier)
        mutator_index = len(pipeline.mutators) + 1
        name = f"{pipeline.name}.{mutator_index}.{component.name}.{mutator_name}"
        super().__init__(name, component, required_resources)

        self._pipeline = pipeline
        self._source = modifier

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._source(*args, **kwargs)


class Pipeline(Resource):
    """A tool for building up values across several components.

    Pipelines are lazily initialized so that we don't have to put constraints
    on the order in which components are created and set up. The values manager
    will configure a pipeline (set all of its attributes) when the pipeline
    source is created.

    As long as a pipeline is not actually called in a simulation, it does not
    need a source or to be configured. This might occur when writing
    generic components that create a set of pipeline modifiers for
    values that won't be used in the particular simulation.

    Notes
    -----
    Pipelines are highy generic and can be used to calculate values of any type
    through a simulation. *Most* pipelines are intended to calculate simulant
    attributes; for those, use :class:`~vivarium.framework.values.pipeline.AttributePipeline`.
    """

    RESOURCE_TYPE = "value"
    """The type of the resource."""

    def __init__(self, name: str, component: Component | None = None) -> None:
        super().__init__(name, component=component)

        self.source: ValueSource = MissingValueSource(self)
        """The callable source of the value represented by the pipeline."""
        self.mutators: list[ValueModifier] = []
        """A list of callables that directly modify the pipeline source or
        contribute portions of the value."""
        self._combiner: ValueCombiner | None = None
        self.post_processor: list[PostProcessor] = []
        """A list of the transformations to perform in order on the combined output of
        the source and mutators."""
        self._manager: ValuesManager | None = None

    def _get_attr_error(self, attribute: str) -> str:
        return (
            f"The pipeline for {self.name} has no {attribute}. This likely means "
            f"you are attempting to modify a value that hasn't been created."
        )

    def _set_attr_error(self, attribute: str, new_value: Any) -> str:
        current_value = getattr(self, f"_{attribute}")
        return (
            f"A second component is attempting to set the {attribute} for pipeline {self.name} "
            f"with {new_value}, but it already has a {attribute}: {current_value}."
        )

    def _get_property(self, property: T | None, property_name: str) -> T:
        if property is None:
            raise DynamicValueError(self._get_attr_error(property_name))
        return property

    @property
    def combiner(self) -> ValueCombiner:
        """A strategy for combining the source and mutator values into the
        final value represented by the pipeline."""
        return self._get_property(self._combiner, "combiner")

    @property
    def manager(self) -> ValuesManager:
        """A reference to the simulation values manager."""
        return self._get_property(self._manager, "manager")

    def __call__(
        self,
        *args: Any,
        mode: Literal["default", "source", "no-post-processors"] = "default",
        **kwargs: Any,
    ) -> Any:
        """Generates the value represented by this pipeline.

        Arguments
        ---------
        mode
            The mode for pipeline evaluation. One of "default", "source",
            or "no-post-processors".
        args, kwargs
            Pipeline arguments. These should be the arguments to the
            callable source of the pipeline.

        Returns
        -------
            The value represented by the pipeline.

        Raises
        ------
        DynamicValueError
            If the pipeline is invoked without a source set.
        """
        return self._call(*args, mode=mode, **kwargs)

    def _call(
        self,
        *args: Any,
        mode: Literal["default", "source", "no-post-processors"] = "default",
        **kwargs: Any,
    ) -> Any:
        if not self.source:
            raise DynamicValueError(
                f"The dynamic value pipeline for {self.name} has no source. This likely means "
                f"you are attempting to modify a value that hasn't been created."
            )
        value = self.source(*args, **kwargs)
        if mode != "source":
            for mutator in self.mutators:
                value = self.combiner(value, mutator, *args, **kwargs)
        if mode == "default":
            for processor in self.post_processor:
                value = processor(value, self.manager)
        if isinstance(value, pd.Series):
            value.name = self.name

        return value

    def __repr__(self) -> str:
        return f"_Pipeline({self.name})"

    def __hash__(self) -> int:
        return hash(self.name)

    def get_value_modifier(
        self,
        modifier: Callable[..., Any],
        component: Component | Manager,
        required_resources: Iterable[str | Resource],
    ) -> ValueModifier:
        """Adds a value modifier to the pipeline and returns it.

        Parameters
        ----------
        modifier
            The value modifier callable for the ValueModifier.
        component
            The component that creates the value modifier.
        required_resources
            A list of resources required by the modifier. A string represents a population attribute.
        """
        value_modifier = ValueModifier(self, modifier, component, required_resources)
        self.mutators.append(value_modifier)
        self._required_resources = [*self._required_resources, value_modifier]
        return value_modifier

    def set_attributes(
        self,
        component: Component | Manager,
        source: ValueSource,
        combiner: ValueCombiner,
        post_processor: list[PostProcessor],
        required_resources: Iterable[str | Resource],
        manager: ValuesManager,
    ) -> None:
        """
        Adds a source, combiner, post-processor, and manager to a pipeline.

        Parameters
        ----------
        component
            The component that creates the pipeline.
        source
            The source for the dynamic attribute pipeline. This can be a callable
            or a list of column names. If a list of column names is provided,
            the component that is registering this attribute producer must be the
            one that creates those columns.
        combiner
            A strategy for combining the source and mutator values into the
            final value represented by the pipeline.
        post_processor
            An optional final transformation to perform on the combined output
            of the source and mutators.
        required_resources
            A list of resources required by the pipeline source, combiner, and
            post-processor. A string represents a population attribute.
        manager
            The simulation values manager.

        Raises
        ------
        DynamicValueError
            If a second component attempts to set the source for a pipeline that
            already has a source.
        """
        if self.source:
            raise DynamicValueError(
                f"A second component is attempting to set the source for pipeline {self.name} "
                f"with {source}, but it already has a source: {self.source}."
            )

        self._component = component
        self.source = source
        self._combiner = combiner
        self.post_processor = post_processor
        self._required_resources = [*self._required_resources, *required_resources]
        self._manager = manager


class AttributePipeline(Pipeline):
    """A type of value pipeline for calculating simulant attributes.

    An attribute pipeline is a specific type of :class:`~vivarium.framework.values.pipeline.Pipeline`
    where the source and callable must take a pd.Index of integers and return a pd.Series
    or pd.DataFrame that has that same index.

    """

    RESOURCE_TYPE = "attribute"
    """The type of the resource."""

    @property
    def is_simple(self) -> bool:
        """Whether or not this ``AttributePipeline`` is simple, i.e. it has a list
        of columns as its source and no modifiers or postprocessors."""
        return (
            isinstance(self.source, PrivateColumnValueSource)
            and not self.mutators
            and not self.post_processor
        )

    def __init__(self, name: str, component: Component | None = None) -> None:
        super().__init__(name, component=component)
        # Re-define the post-processor type to be more specific
        self.post_processor: list[AttributePostProcessor] = []  # type: ignore[assignment]
        """A list of the transformations to perform in order on the combined output of
        the source and mutators."""

    def __call__(  # type: ignore[override]
        self,
        index: pd.Index[int],
        mode: Literal["default", "source", "no-post-processors"] = "default",
    ) -> pd.Series[Any] | pd.DataFrame:
        """Generates the attributes represented by this pipeline.

        Arguments
        ---------
        index
            A pd.Index of integers representing the simulants for which we
            want to calculate the attribute.
        mode
            The mode for pipeline evaluation. One of "default", "source",
            or "no-post-processors".

        Returns
        -------
            A pd.Series or pd.DataFrame of attributes for the simulants in `index`.

        Raises
        ------
        DynamicValueError
            If the pipeline is invoked without a source set.
        """
        # NOTE: must pass index in as arg (NOT kwarg!) to match signature of parent Pipeline._call()
        # Always skip post-processor at _call level; AttributePipeline handles it here.
        # Pass "source" mode through so _call also skips mutators when needed.
        _call_mode: Literal["source", "no-post-processors"] = (
            "source" if mode == "source" else "no-post-processors"
        )
        attribute = self._call(index, mode=_call_mode)
        if mode == "default":
            for processor in self.post_processor:
                attribute = processor(index, attribute, self.manager)
        if not isinstance(attribute, (pd.Series, pd.DataFrame)):
            raise DynamicValueError(
                f"The dynamic attribute pipeline for {self.name} returned a {type(attribute)} "
                "but pd.Series' or pd.DataFrames are expected for attribute pipelines."
            )
        if not attribute.index.equals(index):
            raise DynamicValueError(
                f"The dynamic attribute pipeline for {self.name} returned a series "
                "or dataframe with a different index than was passed in. "
                f"\nReturned index: {attribute.index}"
                f"\nExpected index: {index}"
            )
        return attribute

    def __repr__(self) -> str:
        return f"_AttributePipeline({self.name})"
