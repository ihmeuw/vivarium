# mypy: ignore-errors
"""
=========
Observers
=========

An observer is a component that is responsible for registering
:class:`observations <vivarium.framework.results.observation.BaseObservation>`
to the simulation.

The provided :class:`Observer` class is an abstract base class that should be subclassed
by concrete observers. Each concrete observer is required to implement a
`register_observations` method that registers all required observations.

"""

from abc import ABC, abstractmethod
from typing import Any, Dict

from vivarium import Component
from vivarium.framework.engine import Builder


class Observer(Component, ABC):
    """An abstract base class intended to be subclassed by observer components.

    Notes
    -----
        A `register_observation` method must be defined in the subclass.

    """

    def __init__(self) -> None:
        super().__init__()
        self.results_dir = None

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.get_configuration_name(): {
                    "exclude": [],
                    "include": [],
                },
            },
        }

    def get_configuration_name(self) -> str:
        """Return the name of a concrete observer for use in the configuration"""
        return self.name.split("_observer")[0]

    @abstractmethod
    def register_observations(self, builder: Builder) -> None:
        """(Required). Register observations with within each observer."""
        pass

    def setup_component(self, builder: Builder) -> None:
        """Set up the observer component."""
        super().setup_component(builder)
        self.register_observations(builder)
        self.set_results_dir(builder)

    def set_results_dir(self, builder: Builder) -> None:
        """Define the results directory from the configuration."""
        self.results_dir = (
            builder.configuration.to_dict()
            .get("output_data", {})
            .get("results_directory", None)
        )
