import re
from abc import ABC, abstractmethod
from inspect import signature
from typing import TYPE_CHECKING, Callable, List, Optional

from vivarium.framework.event import Event

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import PopulationView, SimulantData


class Component(ABC):
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
    def name(self) -> str:
        """
        The name of the component. By convention these are in snake case with
        arguments of the `__init__` appended separated by '.'.

        Names must be unique within a simulation.
        """
        if not self._name:
            name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", self.__class__.__name__)
            name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()

            # This is making the assumption that all arguments to the `__init__`
            # have been saved in an attribute of the same name.
            args = [
                str(self.__getattribute__(x)) for x in signature(self.__init__).parameters
            ]
            name = ".".join([name] + args)
            self._name = name

        return self._name

    @property
    def sub_components(self) -> List["Component"]:
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
        """The priority of this component's time-step listener if it exists."""
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
        self._name: str = ""
        self._sub_components: List["Component"] = []
        self.population_view: Optional[PopulationView] = None

    def setup(self, builder: "Builder") -> None:
        """Method that vivarium will run during the setup phase."""
        self.set_population_view(builder)
        self.register_simulant_initializer(builder)
        self.register_time_step_prepare_listener(builder)
        self.register_time_step_listener(builder)
        self.register_time_step_cleanup_listener(builder)
        self.register_collect_metrics_listener(builder)

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
        if self.on_initialize_simulants is not None:
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
        if self.on_time_step_prepare is not None:
            builder.event.register_listener(
                "time_step__prepare",
                self.on_time_step_prepare,
                self.time_step_prepare_priority,
            )

    def register_time_step_listener(self, builder: "Builder") -> None:
        """Registers a time-step listener if this component has defined one."""
        if self.on_time_step is not None:
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
        if self.on_time_step_cleanup is not None:
            builder.event.register_listener(
                "time_step__cleanup",
                self.on_time_step_cleanup,
                self.time_step_cleanup_priority,
            )

    def register_collect_metrics_listener(self, builder: "Builder") -> None:
        """
        Registers a collect metrics listener if this component has defined one.
        """
        if self.on_collect_metrics is not None:
            builder.event.register_listener(
                "collect_metrics",
                self.on_collect_metrics,
                self.collect_metrics_priority,
            )

    ########################
    # Event-driven methods #
    ########################

    on_initialize_simulants: Optional[Callable[["SimulantData"], None]] = None
    on_time_step_prepare: Optional[Callable[[Event], None]] = None
    on_time_step: Optional[Callable[[Event], None]] = None
    on_time_step_cleanup: Optional[Callable[[Event], None]] = None
    on_collect_metrics: Optional[Callable[[Event], None]] = None
