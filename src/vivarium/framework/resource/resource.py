from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING, Any

from vivarium.framework.lifecycle import LifeCycleError

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.framework.population import SimulantData
    from vivarium.manager import Manager


class ResourceId(str):
    """A string representing the unique identifier for a resource, including its type."""

    resource_type: str
    name: str

    def __new__(cls, resource_type: str, name: str) -> ResourceId:
        instance = super().__new__(cls, f"{resource_type}.{name}")
        instance.resource_type = resource_type
        instance.name = name
        return instance

    def __repr__(self) -> str:
        return f"ResourceId({self.resource_type!r}, {self.name!r})"


class Resource:
    """A generic resource representing a node in the dependency graph."""

    RESOURCE_TYPE = "generic_resource"
    """The type of the resource. Should be overridden by subclasses."""

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
        self.on_dependencies_changed: Callable[[], None] | None = None
        """Optional callback invoked when this resource's dependencies change.
        Set by the ResourceManager when the resource is registered."""
        self._raw_required_resources: list[str | Resource] = list(required_resources)
        """The resources required to produce this resource. A string is interpreted
        as the name of an AttributePipeline resource."""

    @property
    def _required_resources(self) -> list[str | Resource]:
        """The raw required resources list. Setting this calls the dependency change callback."""
        return self._raw_required_resources

    @_required_resources.setter
    def _required_resources(self, value: Iterable[str | Resource]) -> None:
        new_value = list(value)
        if new_value != self._raw_required_resources:
            self._raw_required_resources = new_value
            self.notify_dependencies_changed()

    def notify_dependencies_changed(self) -> None:
        """Notify the resource manager that this resource's dependencies have changed."""
        if self.on_dependencies_changed is not None:
            self.on_dependencies_changed()

    @property
    def component(self) -> Component | Manager:
        """The component that creates the resource."""
        if self._component is None:
            raise LifeCycleError(
                f"The component for the resource '{self.resource_id}' has not been set yet."
            )
        return self._component

    @property
    def resource_id(self) -> ResourceId:
        """The long name of the resource, including the type."""
        return ResourceId(self.RESOURCE_TYPE, self.name)

    @property
    def required_resources(self) -> list[ResourceId]:
        """The long names (including type) of required resources for this group."""
        from vivarium.framework.values import AttributePipeline

        return [
            dep.resource_id
            if isinstance(dep, Resource)
            else AttributePipeline.get_resource_id(dep)
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

    @classmethod
    def get_resource_id(cls, name: str) -> ResourceId:
        """Get a resource id for a resource with the given name and this resource's type."""
        return ResourceId(cls.RESOURCE_TYPE, name)


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

    RESOURCE_TYPE = "initializer"
    """The type of the resource."""


class Column(Resource):
    """A resource representing a column in the population private data."""

    def __init__(
        self,
        name: str,
        component: Component | Manager,
        required_resources: Iterable[str | Resource] = (),
    ) -> None:
        super().__init__(name, component, required_resources)

    RESOURCE_TYPE = "column"
    """The type of the resource."""

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Column)
            and self.resource_id == value.resource_id
            and self.component.name == value.component.name
        )

    def __hash__(self) -> int:
        return hash((self.resource_id, self.component.name))
