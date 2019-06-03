"""
==========================
Vivarium Interactive Tools
==========================

This module provides an interface for interactive simulation usage. The main
part is the :class:`InteractiveContext`, a sub-class of the main simulation
object in ``vivarium`` that has been extended to include convenience
methods for running and exploring the simulation in an interactive setting.

Rather than initializing the :class:`InteractiveContext` directly, users should
acquire one using one of the four simulation creation
:ref:`functions <simulation_creation>` provided
in this module.

See the associated tutorials for :ref:`running <interactive_tutorial>` and
:ref:`exploring <exploration_tutorial>` for more information.

"""
from math import ceil
from typing import Mapping, List, Callable, Dict, Any
import warnings

import pandas as pd

from vivarium.framework.configuration import build_simulation_configuration, build_model_specification, ConfigTree
from vivarium.framework.engine import SimulationContext
from vivarium.framework.plugins import PluginManager
from vivarium.framework.time import Timedelta, Time
from vivarium.framework.values import Pipeline

from .utilities import run_from_ipython, log_progress, raise_if_not_setup


class InteractiveContext(SimulationContext):
    """A simulation context with helper methods for running simulations
    interactively.

    This class should not be instantiated directly. It should be created with a
    call to one of the helper methods provided in this module.
    """

    def __init__(self, configuration: ConfigTree, components: List, plugin_manager: PluginManager = None):
        super().__init__(configuration, components, plugin_manager)
        self._initial_population = None

    def setup(self):
        """Setup the simulation and initialize its population."""
        super().setup()
        self._start_time = self.clock.time
        self.initialize_simulants()

    def initialize_simulants(self):
        """Initialize this simulation's population.

        This method should be called by the framework, not by the user.
        """
        super().initialize_simulants()
        self._initial_population = self.population.get_population(True)

    @raise_if_not_setup(system_type='run')
    def reset(self):
        """Reset the simulation to its initial state."""
        warnings.warn("This reset method is very crude.  It should work for "
                      "many simple simulations, but we make no guarantees. In "
                      "particular, if you have components that manage their "
                      "own state in any way, this might not work.")
        self.population._population = self._initial_population
        self.clock._time = self._start_time

    @raise_if_not_setup(system_type='run')
    def run(self, with_logging: bool = True) -> int:
        """Run the simulation for the duration specified in the configuration.

        Parameters
        ----------
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        return self.run_until(self.clock.stop_time, with_logging=with_logging)

    @raise_if_not_setup(system_type='run')
    def run_for(self, duration: Timedelta, with_logging: bool = True) -> int:
        """Run the simulation for the given time duration.

        Parameters
        ----------
        duration
            The length of time to run the simulation for. Should be the same
            type as the simulation clock's step size (usually a pandas
            Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

        Returns
        -------
            The number of steps the simulation took.
        """
        return self.run_until(self.clock.time + duration, with_logging=with_logging)

    @raise_if_not_setup(system_type='run')
    def run_until(self, end_time: Time, with_logging: bool = True) -> int:
        """Run the simulation until the provided end time.

        Parameters
        ----------
        end_time
            The time to run the simulation until. The simulation will run until
            its clock is greater than or equal to the provided end time.
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.

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
    def step(self, step_size: Timedelta = None):
        """Advance the simulation one step.

        Parameters
        ----------
        step_size
            An optional size of step to take. Must be the same type as the
            simulation clock's step size (usually a pandas.Timedelta).
        """
        old_step_size = self.clock.step_size
        if step_size is not None:
            if not isinstance(step_size, type(self.clock.step_size)):
                raise ValueError(f"Provided time must be an instance of {type(self.clock.step_size)}")
            self.clock._step_size = step_size
        super().step()
        self.clock._step_size = old_step_size

    @raise_if_not_setup(system_type='run')
    def take_steps(self, number_of_steps: int = 1, step_size: Timedelta = None, with_logging: bool = True):
        """Run the simulation for the given number of steps.

        Parameters
        ----------
        number_of_steps
            The number of steps to take.
        step_size
            An optional size of step to take. Must be the same type as the
            simulation clock's step size (usually a pandas.Timedelta).
        with_logging
            Whether or not to log the simulation steps. Only works in an ipython
            environment.
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
    def get_population(self, untracked: bool = False) -> pd.DataFrame:
        """Get a copy of the population state table.

        Parameters
        ----------
        untracked
            Whether or not to return simulants who are no longer being tracked
            by the simulation.
        """
        return self.population.get_population(untracked)

    @raise_if_not_setup(system_type='value')
    def list_values(self) -> List[str]:
        """List the names of all pipelines in the simulation."""
        return list(self.values.keys())

    @raise_if_not_setup(system_type='value')
    def get_value(self, value_pipeline_name: str) -> Pipeline:
        """Get the value pipeline associated with the given name."""
        return self.values.get_value(value_pipeline_name)

    @raise_if_not_setup(system_type='event')
    def list_events(self) -> List[str]:
        """List all event types registered with the simulation."""
        return self.events.list_events()

    @raise_if_not_setup(system_type='event')
    def get_listeners(self, event_type: str) -> List[Callable]:
        """Get all listeners of a particular type of event.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        """
        if event_type not in self.events:
            raise ValueError(f'No event {event_type} in system.')
        return self.events.get_listeners(event_type)

    @raise_if_not_setup(system_type='event')
    def get_emitter(self, event_type: str) -> Callable:
        """Get the callable that emits the given type of events.

        Available event types can be found by calling
        :func:`InteractiveContext.list_events`.

        Parameters
        ----------
        event_type
            The type of event to grab the listeners for.

        """
        if event_type not in self.events:
            raise ValueError(f'No event {event_type} in system.')
        return self.events.get_emitter(event_type)

    @raise_if_not_setup(system_type='component')
    def list_components(self) -> Dict[str, Any]:
        """Get a mapping of component names to components currently in the simulation.

        Returns
        -------
            A dictionary mapping component names to components.

        """
        return  self.component_manager.list_components()

    @raise_if_not_setup(system_type='component')
    def get_component(self, name: str) -> Any:
        """Get the component in the simulation that has ``name``, if present.
        Names are guaranteed to be unique.

        Parameters
        ----------
        name
            A component name.
        Returns
        -------
            A component that has the name ``name`` else None.

        """
        return self.component_manager.get_component(name)


def initialize_simulation(components: List, input_config: Mapping = None,
                          plugin_config: Mapping = None) -> InteractiveContext:
    """Construct a simulation from a list of components, component
    configuration, and a plugin configuration.

    The simulation context returned by this method still needs to be setup by
    calling its setup method. It is mostly useful for testing and debugging.

    Parameters
    ----------
    components
        A list of initialized simulation components. Corresponds to the
        components block of a model specification.
    input_config
        A nested dictionary with any additional simulation configuration
        information needed. Corresponds to the configuration block of a model
        specification.
    plugin_config
        A dictionary containing a description of any simulation plugins to
        include in the simulation. If you're using this argument, you're either
        deep in the process of simulation development or the maintainers have
        done something wrong. Corresponds to the plugins block of a model
        specification.

    Returns
    -------
        An initialized (but not set up) simulation context.
    """
    config = build_simulation_configuration()
    config.update(input_config)
    plugin_manager = PluginManager(plugin_config)

    return InteractiveContext(config, components, plugin_manager)


def setup_simulation(components: List, input_config: Mapping = None,
                     plugin_config: Mapping = None) -> InteractiveContext:
    """Construct a simulation from a list of components and call its setup
    method.

    Parameters
    ----------
    components
        A list of initialized simulation components. Corresponds to the
        components block of a model specification.
    input_config
        A nested dictionary with any additional simulation configuration
        information needed. Corresponds to the configuration block of a model
        specification.
    plugin_config
        A dictionary containing a description of any simulation plugins to
        include in the simulation. If you're using this argument, you're either
        deep in the process of simulation development or the maintainers have
        done something wrong. Corresponds to the plugins block of a model
        specification.

    Returns
    -------
        A simulation context that is setup and ready to run.
    """
    simulation = initialize_simulation(components, input_config, plugin_config)
    simulation.setup()

    return simulation


def initialize_simulation_from_model_specification(model_specification_file: str) -> InteractiveContext:
    """Construct a simulation from a model specification file.

    The simulation context returned by this method still needs to be setup by
    calling its setup method. It is mostly useful for testing and debugging.

    Parameters
    ----------
    model_specification_file
        The path to a model specification file.

    Returns
    -------
        An initialized (but not set up) simulation context.
    """
    model_specification = build_model_specification(model_specification_file)

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    return InteractiveContext(simulation_config, components, plugin_manager)


def setup_simulation_from_model_specification(model_specification_file: str) -> InteractiveContext:
    """Construct a simulation from a model specification file and call its setup
    method.

    Parameters
    ----------
    model_specification_file
        The path to a model specification file.

    Returns
    -------
        A simulation context that is setup and ready to run.
    """
    simulation = initialize_simulation_from_model_specification(model_specification_file)
    simulation.setup()

    return simulation
