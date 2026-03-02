from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vivarium.framework.lifecycle import LifeCycleError
from vivarium.framework.resource.exceptions import ResourceError

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.population import SimulantData
    from vivarium.framework.values import AttributePipeline
    from vivarium.manager import Manager


class Resource:
    """A generic resource representing a node in the dependency graph."""

    def __init__(
        self,
        name: str,
        component: Component | Manager | None,
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        self.name = name
        """The name of the resource."""
        self._component = component
        """The component that creates the resource. Can be None if not yet set."""
        self._required_resources: list[str | Resource] = list(required_resources)
        """The resources required to produce this resource."""

    @property
    def resource_type(self) -> str:
        """The type of the resource. Should be overridden by subclasses."""
        return "generic_resource"

    @property
    def component(self) -> Component | Manager:
        """The component that creates the resource."""
        if self._component is None:
            raise LifeCycleError(
                f"The component for the resource '{self.resource_id}' has not been set yet."
            )
        return self._component

    @property
    def resource_id(self) -> str:
        """The long name of the resource, including the type."""
        return f"{self.resource_type}.{self.name}"

    @property
    def required_resources(self) -> list[str]:
        """The long names (including type) of required resources for this group."""
        dependency_strings = [dep for dep in self._required_resources if isinstance(dep, str)]
        if dependency_strings:
            raise ResourceError(
                "Resource has not been finalized; required_resources are still strings.\n"
                f"Resource: {self}\n"
                f"String required_resources: {dependency_strings}"
            )
        return [dep.resource_id for dep in self._required_resources]  # type: ignore[union-attr]

    def finalize_resource(self, attribute_pipelines: dict[str, AttributePipeline]) -> None:
        """Converts any required resources specified as strings to
        :class:`AttributePipelines <vivarium.framework.values.pipeline.AttributePipeline>`."""
        self._required_resources = [
            attribute_pipelines[dep] if isinstance(dep, str) else dep
            for dep in self._required_resources
        ]

    @staticmethod
    def get_callable_name(callable_: Callable[..., Any]) -> str:
        """Get reproducible names based on the callable type."""
        if hasattr(callable_, "name"):
            # This is Pipeline or lookup table or something similar
            modifier_name: str = callable_.name
        elif hasattr(callable_, "__name__"):
            # This is a method or a function
            modifier_name = callable_.__name__
        elif hasattr(callable_, "__call__"):
            # Some anonymous callable
            modifier_name = f"{callable_.__class__.__name__}.__call__"
        else:  # I don't know what this is.
            raise ValueError(f"Unknown callable type: {type(callable_)}")
        return modifier_name


class Initializer(Resource):
    """A resource representing a method for initializing simulant state."""

    def __init__(
        self,
        index: int,
        component: Component | Manager,
        initializer: Callable[[SimulantData], None],
        required_resources: Iterable[str | Resource],
    ) -> None:
        name = f"{index}.{component.name}.{self.get_callable_name(initializer)}"
        super().__init__(name, component, required_resources)
        self.initializer = initializer
        """The initializer method that this resource represents."""

    @property
    def resource_type(self) -> str:
        return "initializer"


class Column(Resource):
    """A resource representing a column in the population private data."""

    def __init__(
        self,
        name: str,
        component: Component | Manager,
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        super().__init__(name, component, required_resources)

    @property
    def resource_type(self) -> str:
        return "column"

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Column)
            and self.resource_id == value.resource_id
            and self.component.name == value.component.name
        )

    def __hash__(self) -> int:
        return hash((self.resource_id, self.component.name))
