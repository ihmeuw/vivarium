from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from vivarium.framework.resource import Resource
from vivarium.framework.values.exceptions import DynamicValueError

if TYPE_CHECKING:
    from vivarium.framework.values.combiners import ValueCombiner
    from vivarium.framework.values.manager import ValuesManager
    from vivarium.framework.values.post_processors import PostProcessor

T = TypeVar("T")


class ValueSource(Resource):
    """A resource representing the source of a value pipeline."""

    def __init__(self, name: str) -> None:
        super().__init__("value_source", name)


class MissingValueSource(Resource):
    """A resource representing an undefined source of a value pipeline."""

    def __init__(self, name: str) -> None:
        super().__init__("missing_value_source", name)


class ValueModifier(Resource):
    """A resource representing a modifier of a value pipeline."""

    def __init__(self, name: str) -> None:
        super().__init__("value_modifier", name)


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
    """

    def __init__(self, name: str) -> None:
        super().__init__("value", name)

        self.source: Callable[..., Any] | None = None
        """The callable source of the value represented by the pipeline."""
        self.mutators: list[Callable[..., Any]] = []
        """A list of callables that directly modify the pipeline source or
        contribute portions of the value."""
        self._combiner: ValueCombiner | None = None
        self.post_processor: PostProcessor | None = None
        """An optional final transformation to perform on the combined output of
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

    def __call__(self, *args: Any, skip_post_processor: bool = False, **kwargs: Any) -> Any:
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

    def _call(self, *args: Any, skip_post_processor: bool = False, **kwargs: Any) -> Any:
        if not self.source:
            raise DynamicValueError(
                f"The dynamic value pipeline for {self.name} has no source. This likely means "
                f"you are attempting to modify a value that hasn't been created."
            )
        value = self.source(*args, **kwargs)
        for mutator in self.mutators:
            value = self.combiner(value, mutator, *args, **kwargs)
        if self.post_processor and not skip_post_processor:
            return self.post_processor(value, self.manager)
        if isinstance(value, pd.Series):
            value.name = self.name

        return value

    def __repr__(self) -> str:
        return f"_Pipeline({self.name})"

    @classmethod
    def setup_pipeline(
        cls,
        pipeline: Pipeline,
        source: Callable[..., Any],
        combiner: ValueCombiner,
        post_processor: PostProcessor | None,
        manager: ValuesManager,
    ) -> None:
        """
        Add a source, combiner, and post-processor to a pipeline.

        Parameters
        ----------
        pipeline
            The pipeline to configure.
        source
            The callable source of the value represented by the pipeline.
        combiner
            A strategy for combining the source and mutator values into the
            final value represented by the pipeline.
        post_processor
            An optional final transformation to perform on the combined output
            of the source and mutators.
        manager
            The simulation values manager.
        """
        pipeline.source = source
        pipeline._combiner = combiner
        pipeline.post_processor = post_processor
        pipeline._manager = manager
