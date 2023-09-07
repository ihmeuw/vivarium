import re
from abc import ABC
from inspect import signature
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from vivarium.framework.event import Event

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import PopulationView, SimulantData


DEFAULT_EVENT_PRIORITY = 5


class Component(ABC):
    """
    A component that can be used in a Vivarium simulation. This component is
    expected to be sub-classed to create components for use in Vivarium
    simulations.

    If a Component takes arguments to its `__init__` function, those arguments
    are expected to have been saved as attributes with the same name as the
    argument in the `__init__` signature.
    """

    """
    A dictionary containing the defaults for any configurations managed by this
    component.
    """
    CONFIGURATION_DEFAULTS: Dict[str, Any] = {}

    def __repr__(self):
        """A string representation of the __init__ call made to create this object"""
        if not self._repr:
            args = [
                f"{name}={value.__repr__() if isinstance(value, Component) else value}"
                for name, value in self.get_initialization_parameters().items()
            ]
            args = ", ".join(args)
            self._repr = f"{type(self).__name__}({args})"

        return self._repr

    def __str__(self):
        return self._repr

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
            base_name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", type(self).__name__)
            base_name = re.sub("([a-z0-9])([A-Z])", r"\1_\2", base_name).lower()

            args = [
                f"'{value.name}'" if isinstance(value, Component) else str(value)
                for value in self.get_initialization_parameters().values()
            ]
            self._name = ".".join([base_name] + args)

        return self._name

    @property
    def sub_components(self) -> List["Component"]:
        """A list of the components that are managed by this component."""
        return self._sub_components

    @property
    def configuration_defaults(self) -> Dict[str, Any]:
        """
        A dictionary containing the defaults for any configurations managed by
        this component.
        """
        return self.CONFIGURATION_DEFAULTS

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
    def initialization_requirements(self) -> Dict[str, List[str]]:
        """
        A dict containing the values this component requires during simulant
        initialization in addition to the columns it creates.

        For each key, an empty list means nothing new is needed during
        initialization.
        """
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [],
        }

    @property
    def population_view_query(self) -> Optional[str]:
        """
        A pandas query string to use to filter the component's `PopulationView`.

        None if no filtering is desired.
        """
        return None

    @property
    def post_setup_priority(self) -> int:
        """The priority of this component's post_setup listener if it exists."""
        return DEFAULT_EVENT_PRIORITY

    @property
    def time_step_prepare_priority(self) -> int:
        """
        The priority of this component's time_step__prepare listener if it
        exists.
        """
        return DEFAULT_EVENT_PRIORITY

    @property
    def time_step_priority(self) -> int:
        """The priority of this component's time_step listener if it exists."""
        return DEFAULT_EVENT_PRIORITY

    @property
    def time_step_cleanup_priority(self) -> int:
        """The priority of this component's time_step__cleanup listener if it exists."""
        return DEFAULT_EVENT_PRIORITY

    @property
    def collect_metrics_priority(self) -> int:
        """
        The priority of this component's collect_metrics listener if it exists.
        """
        return DEFAULT_EVENT_PRIORITY

    @property
    def simulation_end_priority(self) -> int:
        """The priority of this component's simulation_end listener if it exists."""
        return DEFAULT_EVENT_PRIORITY

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self):
        self._repr: str = ""
        self._name: str = ""
        self._sub_components: List["Component"] = []
        self.logger = None
        self.population_view: Optional[PopulationView] = None

    def setup(self, builder: "Builder") -> None:
        """Method that vivarium will run during the setup phase."""
        self.logger = builder.logging.get_logger(self.name)
        self.set_population_view(builder)
        self.register_post_setup_listener(builder)
        self.register_simulant_initializer(builder)
        self.register_time_step_prepare_listener(builder)
        self.register_time_step_listener(builder)
        self.register_time_step_cleanup_listener(builder)
        self.register_collect_metrics_listener(builder)
        self.register_simulation_end_listener(builder)

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
            self.population_view = builder.population.get_view(
                population_view_columns, self.population_view_query
            )

    def register_post_setup_listener(self, builder: "Builder") -> None:
        """Registers a post_setup listener if this component has defined one."""
        if type(self).on_post_setup != Component.on_post_setup:
            builder.event.register_listener(
                "post_setup",
                self.on_post_setup,
                self.post_setup_priority,
            )

    def register_simulant_initializer(self, builder: "Builder") -> None:
        """Registers a simulant initializer if this component has defined one."""
        if type(self).on_initialize_simulants != Component.on_initialize_simulants:
            builder.population.initializes_simulants(
                self.on_initialize_simulants,
                creates_columns=self.columns_created,
                **self.initialization_requirements,
            )

    def register_time_step_prepare_listener(self, builder: "Builder") -> None:
        """
        Registers a time_step__prepare listener if this component has defined
        one.
        """
        if type(self).on_time_step_prepare != Component.on_time_step_prepare:
            builder.event.register_listener(
                "time_step__prepare",
                self.on_time_step_prepare,
                self.time_step_prepare_priority,
            )

    def register_time_step_listener(self, builder: "Builder") -> None:
        """Registers a time_step listener if this component has defined one."""
        if type(self).on_time_step != Component.on_time_step:
            builder.event.register_listener(
                "time_step",
                self.on_time_step,
                self.time_step_priority,
            )

    def register_time_step_cleanup_listener(self, builder: "Builder") -> None:
        """
        Registers a time_step__cleanup listener if this component has defined
        one.
        """
        if type(self).on_time_step_cleanup != Component.on_time_step_cleanup:
            builder.event.register_listener(
                "time_step__cleanup",
                self.on_time_step_cleanup,
                self.time_step_cleanup_priority,
            )

    def register_collect_metrics_listener(self, builder: "Builder") -> None:
        """
        Registers a collect_metrics listener if this component has defined one.
        """
        if type(self).on_collect_metrics != Component.on_collect_metrics:
            builder.event.register_listener(
                "collect_metrics",
                self.on_collect_metrics,
                self.collect_metrics_priority,
            )

    def register_simulation_end_listener(self, builder: "Builder") -> None:
        """Registers a post_setup listener if this component has defined one."""
        if type(self).on_simulation_end != Component.on_simulation_end:
            builder.event.register_listener(
                "simulation_end",
                self.on_simulation_end,
                self.simulation_end_priority,
            )

    ########################
    # Event-driven methods #
    ########################

    def on_post_setup(self, event: Event) -> None:
        """
        Method that vivarium will run during the post_setup event.

        NOTE: this is not commonly used functionality.
        """
        pass

    def on_initialize_simulants(self, pop_data: "SimulantData") -> None:
        """Method that vivarium will run during simulant initialization"""
        pass

    def on_time_step_prepare(self, event: Event) -> None:
        """Method that vivarium will run during the time_step__prepare event"""
        pass

    def on_time_step(self, event: Event) -> None:
        """Method that vivarium will run during the time_step event"""
        pass

    def on_time_step_cleanup(self, event: Event) -> None:
        """Method that vivarium will run during the time_step__cleanup event"""
        pass

    def on_collect_metrics(self, event: Event) -> None:
        """Method that vivarium will run during the collect_metrics event"""
        pass

    def on_simulation_end(self, event: Event) -> None:
        """Method that vivarium will run during the simulation_end event."""
        pass

    ##################
    # Helper methods #
    ##################

    def get_initialization_parameters(self) -> Dict[str, Any]:
        """
        Gets the values of all parameters specified in the __init__` that have
        an attribute with the same name.

        This makes the assumption that arguments to the `__init__` are saved as
        attributes with the same name.

        Note: this retrieves the value of the attribute at the time of calling
        which is not guaranteed to be the same as the original value.
        """

        return {
            parameter_name: getattr(self, parameter_name)
            for parameter_name in signature(self.__init__).parameters
            if hasattr(self, parameter_name)
        }
