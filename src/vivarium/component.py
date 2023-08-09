from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional, Callable

from vivarium.framework.event import Event

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData, PopulationView


class VivariumComponent(ABC):
    """A component that can be used in a Vivarium simulation."""

    """
    A dictionary containing the defaults for any configurations managed by this
    component.
    """
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
        """
        A list of the columns this component creates.

        An empty list means this component does not create any columns.
        """
        return []

    @property
    def columns_required(self) -> Optional[List[str]]:
        """
        A list of the columns this component requires that it did not create.

        An empty list means all available columns are required.
        `None` means no additional columns are required.
        """
        return None

    @property
    def initialization_columns_required(self) -> List[str]:
        """
        A list of the columns this component requires during simulant
        initialization.

        An empty list means no columns beyond those created by this component
        are needed during initialization.
        """
        return []

    @property
    def time_step_prepare_priority(self) -> int:
        """
        The priority of this component's time-step prepare listener if it
        exists.
        """
        return 5

    @property
    def time_step_priority(self) -> int:
        """ The priority of this component's time-step listener if it exists. """
        return 5

    @property
    def time_step_cleanup_priority(self) -> int:
        """
        The priority of this component's time-step cleanup listener if it
        exists.
        """
        return 5

    @property
    def collect_metrics_priority(self) -> int:
        """
        The priority of this component's collect metrics listener if it exists.
        """
        return 5

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        self._sub_components: List["VivariumComponent"] = []
        self.population_view: Optional[PopulationView] = None

    def setup(self, builder: "Builder") -> None:
        """Method that vivarium will run during the setup phase."""
        self.set_population_view(builder)

    def on_post_setup(self, builder: "Builder") -> None:
        """
        Method that vivarium will run during the post-setup phase.

        NOTE: this is not commonly used functionality.
        """
        pass

    #################
    # Setup methods #
    #################

    def set_population_view(self, builder: "Builder") -> None:
        """
        Creates the PopulationView for this component if it needs access to the
        state table.
        """

        if self.columns_required:
            # Get all columns created and required
            population_view_columns = self.columns_created + self.columns_required
        elif self.columns_required == []:
            # Empty list means population view needs all available columns
            population_view_columns = []
        elif self.columns_required is None and self.columns_created:
            # No additional columns required, so just get columns created
            population_view_columns = self.columns_created
        else:
            # no need for a population view if no columns created or required
            population_view_columns = None

        if population_view_columns is not None:
            self.population_view = builder.population.get_view(population_view_columns)

    def register_simulant_initializer(self, builder: "Builder") -> None:
        """Registers a simulant initializer if this component has defined one."""
        builder.population.initializes_simulants(
            self.on_initialize_simulants,
            creates_columns=self.columns_created,
            requires_streams=self.initialization_columns_required,
        )

    def register_time_step_prepare_listener(self, builder: "Builder") -> None:
        """
        Registers a time-step prepare listener if this component has defined
        one.
        """
        builder.event.register_listener(
            "time_step__prepare",
            self.on_time_step_prepare,
            self.time_step_prepare_priority,
        )

    def register_time_step_listener(self, builder: "Builder") -> None:
        """ Registers a time-step listener if this component has defined one. """
        builder.event.register_listener(
            "time_step",
            self.on_time_step,
            self.time_step_priority,
        )

    def register_time_step_cleanup_listener(self, builder: "Builder") -> None:
        """
        Registers a time-step cleanup listener if this component has defined
        one.
        """
        builder.event.register_listener(
            "time_step__cleanup",
            self.on_time_step_cleanup,
            self.time_step_cleanup_priority,
        )

    def register_collect_metrics_listener(self, builder: "Builder") -> None:
        """
        Registers a collect metrics listener if this component has defined one.
        """
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
