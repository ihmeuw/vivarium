from __future__ import annotations

from vivarium.framework.resource.exceptions import ResourceError

RESOURCE_TYPES = {
    "value",
    "value_source",
    "missing_value_source",
    "value_modifier",
    "column",
    "stream",
}


class Resource:
    """A generic resource.

    These resources may be required to build up the dependency graph.
    """

    def __init__(self, type: str, name: str):
        if type not in RESOURCE_TYPES:
            raise ResourceError(f"Unknown resource type: {type}")

        self.type = type
        """The type of the resource."""
        self.name = name
        """The name of the resource."""

    def __str__(self) -> str:
        return f"{self.type}.{self.name}"
