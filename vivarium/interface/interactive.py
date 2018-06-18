from math import ceil

from vivarium import VivariumError
from vivarium.framework.configuration import build_simulation_configuration, build_model_specification
from vivarium.framework.plugins import PluginManager
from vivarium.framework.engine import SimulationContext

from .utilities import run_from_ipython, log_progress


class InteractiveError(VivariumError):
    """Error raised when the Interactive context is in an inconsistent state."""
    pass


class InteractiveContext(SimulationContext):

    def __init__(self, configuration, components, plugin_manager=None):
        super().__init__(configuration, components, plugin_manager)
        self._initial_population = None
        self._setup = False

    def setup(self):
        super().setup()
        self._start_time = self.clock.time
        self.initialize_simulants()
        self._setup = True

    def initialize_simulants(self):
        super().initialize_simulants()
        self._initial_population = self.population.population

    def reset(self):
        # This is super crude, but should work for a great deal of components.
        self.population._population = self._initial_population
        self.clock._time = self._start_time

    def run(self, with_logging=True):
        if not self._setup:
            raise InteractiveError("Simulation must be setup before it can be run.")

        return self.run_until(self.clock.stop_time, with_logging=with_logging)

    def run_for(self, duration, with_logging=True):
        if not self._setup:
            raise InteractiveError("Simulation must be setup before it can be run.")

        return self.run_until(self.clock.time + duration, with_logging=with_logging)

    def run_until(self, end_time, with_logging=True):
        if not self._setup:
            raise InteractiveError("Simulation must be setup before it can be run.")

        if not isinstance(end_time, type(self.clock.time)):
            raise ValueError(f"Provided time must be an instance of {type(self.clock.time)}")

        iterations = int(ceil((end_time - self.clock.time)/self.clock.step_size))
        self.take_steps(number_of_steps=iterations, with_logging=with_logging)
        assert self.clock.time - self.clock.step_size < end_time <= self.clock.time
        return iterations

    def step(self, step_size=None):  # TODO: consider renaming to take_step for similarity with sim.take_steps
        if not self._setup:
            raise InteractiveError("Simulation must be setup before it can be run.")

        old_step_size = self.clock.step_size
        if step_size is not None:
            if not isinstance(step_size, type(self.clock.step_size)):
                raise ValueError(f"Provided time must be an instance of {type(self.clock.step_size)}")
            self.clock._step_size = step_size
        super().step()
        self.clock._step_size = old_step_size

    def take_steps(self, number_of_steps=1, step_size=None, with_logging=True):
        if not self._setup:
            raise InteractiveError("Simulation must be setup before it can be run.")

        if not isinstance(number_of_steps, int):
            raise ValueError('Number of steps must be an integer.')

        if run_from_ipython() and with_logging:
            for _ in log_progress(range(number_of_steps), name='Step'):
                self.step(step_size)
        else:
            for _ in range(number_of_steps):
                self.step(step_size)

    def list_values(self):
        if not self._setup:
            raise InteractiveError("Value pipeline configuration is not complete until the simulation is setup.")

        return list(self.values.keys())

    def get_values(self):
        if not self._setup:
            raise InteractiveError("Value pipeline configuration is not complete until the simulation is setup.")

        return self.values.items()

    def get_value(self, value_pipeline_name):
        return self.values.get_value(value_pipeline_name)

    def list_events(self):
        if not self._setup:
            raise InteractiveError("Event configuration is not complete until the simulation is setup.")

        return self.events.list_events()

    def get_listeners(self, event_name):
        if not self._setup:
            raise InteractiveError("Event configuration is not complete until the simulation is setup.")

        if event_name not in self.events:
            raise ValueError(f'No event {event_name} in system.')
        return self.events.get_listeners(event_name)

    def get_emitter(self, event_name):
        if not self._setup:
            raise InteractiveError("Event configuration is not complete until the simulation is setup.")

        if event_name not in self.events:
            raise ValueError(f'No event {event_name} in system.')
        return self.events.get_emitter(event_name)

    def get_components(self):
        if not self._setup:
            raise InteractiveError("Component configuration is not complete until the simulation is setup.")

        return [component for component in self.component_manager._components + self.component_manager._managers]

    def reload_component(self, component):
        raise NotImplementedError()

    def replace_component(self, old_component, new_component):
        if not self._setup:
            raise InteractiveError("Components cannot be replaced until the simulation is setup.")

        self.component_manager._components.remove(old_component)
        new_component.setup(self.builder)
        self.component_manager.add_components([new_component])


def initialize_simulation(components, input_config=None):
    config = build_simulation_configuration()
    config.update(input_config)

    return InteractiveContext(config, components)


def setup_simulation(components, input_config=None):
    simulation = initialize_simulation(components, input_config)
    simulation.setup()

    return simulation


def initialize_simulation_from_model_specification(model_specification_file):
    model_specification = build_model_specification(model_specification_file)

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    return InteractiveContext(simulation_config, components, plugin_manager)


def setup_simulation_from_model_specification(model_specification_file):
    simulation = initialize_simulation_from_model_specification(model_specification_file)
    simulation.setup()

    return simulation
