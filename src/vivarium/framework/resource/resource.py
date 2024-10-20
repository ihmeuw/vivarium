from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Resource:
    """A generic resource representing a node in the dependency graph."""

    resource_type: str
    """The type of the resource."""
    name: str
    """The name of the resource."""

    @property
    def resource_id(self) -> str:
        """The long name of the resource, including the type."""
        return f"{self.resource_type}.{self.name}"

    # TODO [MIC_TBD]: make this an abstract method when support for old
    #  requirements specification is dropped
    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return False


class NullResource(Resource):
    """A node in the dependency graph that does not produce any resources."""

    def __init__(self, index: int):
        super().__init__("null", f"{index}")

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True


class Column(Resource):
    """A resource representing a column in the state table."""

    def __init__(self, name: str):
        super().__init__("column", name)

    @property
    def is_initialized(self) -> bool:
        """Return True if the resource needs to be initialized."""
        return True
