from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from vivarium.framework.event import Event

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import PopulationView, SimulantData


class VivariumComponent(ABC):
    """A component that can be used in a Vivarium simulation."""

    configuration_defaults = {}

    @abstractmethod
    def __repr__(self):
        pass

    ##############
    # Properties #
    ##############

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the component. Names must be unique within a simulation."""
        pass

    @property
    def sub_components(self) -> List["VivariumComponent"]:
        """A list of the components that are managed by this component."""
        return self._sub_components

    @property
    def columns_created(self) -> List[str]:
        """A list of the columns this component creates."""
        return []

    @property
    def columns_required(self) -> List[str]:
        """A list of the columns this component requires that it did not create."""
        return []

    @property
    def initialization_columns_required(self) -> List[str]:
        """A list of the columns this component requires during simulant initialization."""
        return []

    @property
    def time_step_prepare_priority(self) -> int:
        return 5

    @property
    def time_step_priority(self) -> int:
        return 5

    @property
    def time_step_cleanup_priority(self) -> int:
        return 5

    @property
    def collect_metrics_priority(self) -> int:
        return 5

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        self._sub_components = []

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: "Builder") -> None:
        self.population_view = self.get_population_view(builder)

    def on_post_setup(self, builder: "Builder") -> None:
        pass

    #################
    # Setup methods #
    #################

    def get_population_view(self, builder: "Builder") -> Optional["PopulationView"]:
        population_view_columns = self.columns_created + self.columns_required
        if population_view_columns:
            return builder.population.get_view(population_view_columns)
        else:
            return None

    def register_simulant_initializer(self, builder: "Builder") -> None:
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_streams=self.initialization_columns_required,
        )

    def register_time_step_prepare_listener(self, builder: "Builder") -> None:
        builder.event.register_listener(
            "time_step__prepare",
            self.on_time_step_prepare,
            self.time_step_prepare_priority,
        )

    def register_time_step_listener(self, builder: "Builder") -> None:
        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            self.time_step_priority,
        )

    def register_time_step_cleanup_listener(self, builder: "Builder") -> None:
        builder.event.register_listener(
            "time_step__cleanup",
            self.on_time_step_cleanup,
            self.time_step_cleanup_priority,
        )

    def register_collect_metrics_listener(self, builder: "Builder") -> None:
        builder.event.register_listener(
            "collect_metrics",
            self.on_collect_metrics,
            self.collect_metrics_priority,
        )

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: "SimulantData") -> None:
        pass

    def on_time_step_prepare(self, event: Event) -> None:
        pass

    def on_time_step(self, event: Event) -> None:
        pass

    def on_time_step_cleanup(self, event: Event) -> None:
        pass

    def on_collect_metrics(self, event: Event) -> None:
        pass
