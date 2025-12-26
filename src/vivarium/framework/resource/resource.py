from __future__ import annotations

from typing import TYPE_CHECKING

from vivarium.framework.lifecycle import LifeCycleError

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.manager import Manager


class Resource:
    """A generic resource representing a node in the dependency graph."""

    def __init__(
        self, resource_type: str, name: str, component: Component | Manager | None
    ) -> None:
        """Create a new resource."""
        self.resource_type = resource_type
        """The type of the resource."""
        self.name = name
        """The name of the resource."""
        self._component = component
        """The component that creates the resource. Can be None if not yet set."""

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
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return False


class NullResource(Resource):
    """A node in the dependency graph that does not produce any resources."""

    def __init__(self, index: int, component: Component | Manager):
        super().__init__("null", f"{index}", component)

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True


class Column(Resource):
    """A resource representing a column in the population private data."""

    def __init__(self, name: str, component: Component | Manager):
        super().__init__("column", name, component)

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, Column)
            and self.resource_id == value.resource_id
            and self.component.name == value.component.name
        )
