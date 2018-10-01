"""An interface for interactive simulation usage."""
from math import ceil
from typing import Collection, List, Callable, Dict
import warnings

import pandas as pd

from vivarium.framework.engine import SimulationContext
from vivarium.framework.configuration import (build_simulation_configuration,
                                              build_model_specification, ConfigTree)
from vivarium.framework.plugins import PluginManager
from vivarium.framework.time import Timedelta, Time
from vivarium.framework.values import Pipeline

from .utilities import run_from_ipython, log_progress, raise_if_not_setup, InteractiveError


class InteractiveContext(SimulationContext):
    """A simulation context with several helper methods for interactive usage.

    This class should not be instantiated directly. It should be created
    with a call to one of the helper methods provided in this module.
    """

    def __init__(self, configuration: ConfigTree, components: Collection, plugin_manager: PluginManager=None):
        super().__init__(configuration, components, plugin_manager)
        self._initial_population = None
        self._setup = False

    def setup(self):
        """Sets up the simulation and initializes its population.

        Should not be called directly.
        """
        super().setup()
        self._start_time = self.clock.time
        self.initialize_simulants()
        self._setup = True

    def initialize_simulants(self):
        """Initializes this simulation's population.

        Should not be called directly
        """
        super().initialize_simulants()
        self._initial_population = self.population.population

    @raise_if_not_setup(system_type='run')
    def reset(self):
        """Reset's this simulation to it's initial state."""
        warnings.warn("This reset method is very crude.  It should work for "
                      "many simple simulations, but we make no guarantees. In "
                      "particular, if you have components that manage their "
                      "own state in any way, this might not work.")
        self.population._population = self._initial_population
        self.clock._time = self._start_time

    @raise_if_not_setup(system_type='run')
    def run(self, with_logging: bool=True) -> int:
        """Runs the simulation for the duration specified in the configuration.

        Parameters
        ----------
        with_logging :
            Whether or not to log the simulation steps. Only works in an
            ipython environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        return self.run_until(self.clock.stop_time, with_logging=with_logging)

    @raise_if_not_setup(system_type='run')
    def run_for(self, duration: Timedelta, with_logging: bool=True) -> int:
        """Runs the simulation for the given duration.

        Parameters
        ----------
        duration :
            The length of time to run for. Should be the same type as the
            simulation clock's step size (usually a pandas.Timedelta).
        with_logging :
            Whether or not to log the simulation steps. Only works in an
            ipython environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        return self.run_until(self.clock.time + duration, with_logging=with_logging)

    @raise_if_not_setup(system_type='run')
    def run_until(self, end_time: Time, with_logging: bool=True) -> int:
        """Runs the simulation until the provided end time.

        Parameters
        ----------
        end_time :
            The time to run until. The simulation will run until it's clock
            is greater than or equal to the provided end time.
        with_logging :
            Whether or not to log the simulation steps. Only works in an
            ipython environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        if not isinstance(end_time, type(self.clock.time)):
            raise ValueError(f"Provided time must be an instance of {type(self.clock.time)}")

        iterations = int(ceil((end_time - self.clock.time)/self.clock.step_size))
        self.take_steps(number_of_steps=iterations, with_logging=with_logging)
        assert self.clock.time - self.clock.step_size < end_time <= self.clock.time
        return iterations

    @raise_if_not_setup(system_type='run')
    def step(self, step_size: Timedelta=None):
        """Takes a single step in the simulation.

        Parameters
        ----------
        step_size :
            An optional size of step to take. Must be the same type as the
            simulation clock's step size, usually a pandas.Timedelta.
        """
        old_step_size = self.clock.step_size
        if step_size is not None:
            if not isinstance(step_size, type(self.clock.step_size)):
                raise ValueError(f"Provided time must be an instance of {type(self.clock.step_size)}")
            self.clock._step_size = step_size
        super().step()
        self.clock._step_size = old_step_size

    @raise_if_not_setup(system_type='run')
    def take_steps(self, number_of_steps: int=1, step_size: Timedelta=None, with_logging: bool=True):
        """Runs the simulation for the given duration.

        Parameters
        ----------
        number_of_steps :
            The number of steps to take
        step_size :
            An optional size of step to take. Must be the same type as the
            simulation clock's step size, usually a pandas.Timedelta.
        with_logging :
            Whether or not to log the simulation steps. Only works in an
            ipython environment.
        """
        if not isinstance(number_of_steps, int):
            raise ValueError('Number of steps must be an integer.')

        if run_from_ipython() and with_logging:
            for _ in log_progress(range(number_of_steps), name='Step'):
                self.step(step_size)
        else:
            for _ in range(number_of_steps):
                self.step(step_size)

    @raise_if_not_setup(system_type='population')
    def get_population(self) -> pd.DataFrame:
        """Gets a copy of the population state table"""
        return self.population.population

    @raise_if_not_setup(system_type='value')
    def list_values(self) -> List[str]:
        """Lists the names of all pipelines in the simulation."""
        return list(self.values.keys())

    @raise_if_not_setup(system_type='value')
    def get_value(self, value_pipeline_name: str) -> Pipeline:
        """Gets the value pipeline associated with the given name."""
        return self.values.get_value(value_pipeline_name)

    @raise_if_not_setup(system_type='event')
    def list_events(self) -> List[str]:
        """Lists all event types registered with the simulation."""
        return self.events.list_events()

    @raise_if_not_setup(system_type='event')
    def get_listeners(self, event_type: str) -> List[Callable]:
        """Gets all listeners to a particular type of event."""
        if event_type not in self.events:
            raise ValueError(f'No event {event_type} in system.')
        return self.events.get_listeners(event_type)

    @raise_if_not_setup(system_type='event')
    def get_emitter(self, event_type: str) -> Callable:
        """Gets a callable that emits the give type of events."""
        if event_type not in self.events:
            raise ValueError(f'No event {event_type} in system.')
        return self.events.get_emitter(event_type)

    @raise_if_not_setup(system_type='component')
    def get_components(self) -> List:
        """Get's a list of all components in the system."""
        return [component for component in self.component_manager._components + self.component_manager._managers]

    def add_components(self, components: List):
        """Adds a list of components to the simulation."""
        if self._setup:
            raise InteractiveError("Can't add components to an already set up simulation.")


def initialize_simulation(components: List, input_config: Dict=None, plugin_config: Dict=None) -> InteractiveContext:
    """Constructs a simulation from a list of components.

    The simulation context constructed here still needs to be setup. The
    context provided here is mostly useful for testing and debugging
    purposes.

    Parameters
    ----------
    components :
        A list of initialized simulation components.
    input_config :
        A nested dictionary with any additional simulation configuration
        information needed.
    plugin_config :
        A dictionary containing a description of any simulation plugins
        to include in the simulation. If you're using this argument, you're
        either deep in the weeds of simulation development, or we've done
        something wrong.

    Returns
    -------
        An initialized (but not set up) simulation context.
    """
    config = build_simulation_configuration()
    config.update(input_config)
    plugin_manager = PluginManager(plugin_config)

    return InteractiveContext(config, components, plugin_manager)


def setup_simulation(components: List, input_config: Dict=None, plugin_config: Dict=None) -> InteractiveContext:
    """Constructs and sets up a simulation from a list of components.

    Parameters
    ----------
    components :
        A list of initialized simulation components.
    input_config :
        A nested dictionary with any additional simulation configuration
        information needed.
    plugin_config :
        A dictionary containing a description of any simulation plugins
        to include in the simulation. If you're using this argument, you're
        either deep in the weeds of simulation development, or we've done
        something wrong.

    Returns
    -------
        A set up simulation context ready to be run.
    """
    simulation = initialize_simulation(components, input_config, plugin_config)
    simulation.setup()

    return simulation


def initialize_simulation_from_model_specification(model_specification_file_path: str) -> InteractiveContext:
    """Constructs a simulation from a model specification file.

    The simulation context constructed here still needs to be setup.

    The primary use case for this function is typically to add new components
    to a pre-existing model represented as a yaml model specification.
    This can be done with
    ``simulation.add_components([component1, component2, ...])``.

    Parameters
    ----------
    model_specification_file_path :
        The path to the model specification.

    Returns
    -------
        An initialized (but not set up) simulation context.
    """
    model_specification = build_model_specification(model_specification_file_path)

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    return InteractiveContext(simulation_config, components, plugin_manager)


def setup_simulation_from_model_specification(model_specification_file_path: str) -> InteractiveContext:
    """Constructs and sets up a simulation from a model specification file.

    Parameters
    ----------
    model_specification_file_path :
        The path to the model specification.

    Returns
    -------
        A set up simulation context ready to be run.
    """
    simulation = initialize_simulation_from_model_specification(model_specification_file_path)
    simulation.setup()

    return simulation
