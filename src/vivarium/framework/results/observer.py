from abc import ABC, abstractmethod
from typing import Any, Dict

from vivarium import Component
from vivarium.framework.engine import Builder


class Observer(Component, ABC):
    """An abstract base class intended to be subclassed by observer components.
    The primary purpose of this class is to provide attributes required by
    the subclass `report` method.

    Note that a `register_observation` method must be defined in the subclass.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results_dir = None
        self.input_draw = None
        self.random_seed = None

    @abstractmethod
    def register_observations(self, builder: Builder) -> None:
        """(Required). Register observations with within each observer."""
        pass

    def setup_component(self, builder: Builder) -> None:
        super().setup_component(builder)
        self.register_observations(builder)
        self.get_report_attributes(builder)

    def get_report_attributes(self, builder: Builder) -> None:
        """Define commonly-used attributes for reporting."""
        self.results_dir = (
            builder.configuration.to_dict()
            .get("output_data", {})
            .get("results_directory", None)
        )
        self.input_draw = (
            builder.configuration.to_dict()
            .get("input_data", {})
            .get("input_draw_number", None)
        )
        self.random_seed = (
            builder.configuration.to_dict().get("randomness", {}).get("random_seed", None)
        )


class StratifiedObserver(Observer):
    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        return {
            "stratification": {
                self.name.split("_observer")[0]: {
                    "exclude": [],
                    "include": [],
                },
            },
        }
