from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vivarium import Component
    from vivarium.manager import Manager


@dataclass
class Resource:
    """A generic resource representing a node in the dependency graph."""

    resource_type: str
    """The type of the resource."""
    name: str
    """The name of the resource."""
    # TODO [MIC-5452]: all resources should have a component
    component: Component | Manager | None
    """The component that creates the resource."""

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

    # TODO [MIC-5452]: all resources should have a component
    def __init__(self, index: int, component: Component | Manager | None):
        super().__init__("null", f"{index}", component)

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True


class Column(Resource):
    """A resource representing a column in the state table."""

    # TODO [MIC-5452]: all resources should have a component
    def __init__(self, name: str, component: Component | Manager | None):
        super().__init__("column", name, component)

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True
