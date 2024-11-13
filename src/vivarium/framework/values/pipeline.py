from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeVar

import pandas as pd

from vivarium import Component
from vivarium.framework.resource import Resource
from vivarium.framework.values.exceptions import DynamicValueError

if TYPE_CHECKING:
    from vivarium.framework.values.combiners import ValueCombiner
    from vivarium.framework.values.manager import ValuesManager
    from vivarium.framework.values.post_processors import PostProcessor

T = TypeVar("T")


class ValueSource(Resource):
    """A resource representing the source of a value pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        source: Callable[..., Any] | None,
        component: Component | None,
    ) -> None:
        super().__init__(
            "value_source" if source else "missing_value_source", pipeline.name, component
        )
        self._pipeline = pipeline
        self._source = source

    def __bool__(self) -> bool:
        return self._source is not None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if not self._source:
            raise DynamicValueError(
                f"The dynamic value pipeline for {self.name} has no source."
                " This likely means you are attempting to modify a value that"
                " hasn't been created."
            )
        return self._source(*args, **kwargs)


class ValueModifier(Resource):
    """A resource representing a modifier of a value pipeline."""

    def __init__(
        self,
        pipeline: Pipeline,
        modifier: Callable[..., Any],
        component: Component | None,
    ) -> None:
        mutator_name = self._get_modifier_name(modifier)
        mutator_index = len(pipeline.mutators) + 1
        name = f"{pipeline.name}.{mutator_index}.{mutator_name}"
        super().__init__("value_modifier", name, component)

        self._pipeline = pipeline
        self._source = modifier

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._source(*args, **kwargs)

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

    def __init__(self, name: str, component: Component | None = None) -> None:
        super().__init__("value", name, component=component)

        self.source: ValueSource = ValueSource(self, source=None, component=None)
        """The callable source of the value represented by the pipeline."""
        self.mutators: list[ValueModifier] = []
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

    def __hash__(self) -> int:
        return hash(self.name)

    def get_value_modifier(
        self, modifier: Callable[..., Any], component: Component | None
    ) -> ValueModifier:
        """Add a value modifier to the pipeline and return it.

        Parameters
        ----------
        modifier
            The value modifier callable for the ValueModifier.
        component
            The component that creates the value modifier.
        """
        value_modifier = ValueModifier(self, modifier, component)
        self.mutators.append(value_modifier)
        return value_modifier

    def set_attributes(
        self,
        component: Component | None,
        source: Callable[..., Any],
        combiner: ValueCombiner,
        post_processor: PostProcessor | None,
        manager: ValuesManager,
    ) -> None:
        """
        Add a source, combiner, post-processor, and manager to a pipeline.

        Parameters
        ----------
        component
            The component that creates the pipeline.
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
        self.component = component
        self.source = ValueSource(self, source, component)
        self._combiner = combiner
        self.post_processor = post_processor
        self._manager = manager
