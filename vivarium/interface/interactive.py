from math import ceil

from vivarium.framework.configuration import build_simulation_configuration, build_model_specification
from vivarium.framework.plugins import PluginManager, DEFAULT_PLUGINS
from vivarium.framework.engine import SimulationContext
from vivarium.config_tree import ConfigTree

from .utilities import run_from_ipython, log_progress


class InteractiveContext(SimulationContext):

    def __init__(self, configuration, components, plugin_manager=None):
        super().__init__(configuration, components, plugin_manager)
        self._initial_population = None

    def setup(self):
        super().setup()
        self._start_time = self.clock.time

    def initialize_simulants(self):
        super().initialize_simulants()
        self._initial_population = self.population.population

    def reset(self):
        # This is super crude, but should work for a great deal of components.
        self.population._population = self._initial_population
        self.clock._time = self._start_time

    def run_for(self, duration, with_logging=True):
        return self.run_until(self.clock.time + duration, with_logging=with_logging)

    def run_until(self, end_time, with_logging=True):
        if not isinstance(end_time, type(self.clock.time)):
            raise ValueError(f"Provided time must be an instance of {type(self.clock.time)}")
        iterations = int(ceil((end_time - self.clock.time)/self.clock.step_size))
        self.take_steps(number_of_steps=iterations, with_logging=with_logging)
        assert self.clock.time - self.clock.step_size < end_time <= self.clock.time
        return iterations

    def step(self, step_size=None):
        old_step_size = self.clock.step_size
        if step_size is not None:
            if not isinstance(step_size, type(self.clock.step_size)):
                raise ValueError(f"Provided time must be an instance of {type(self.clock.step_size)}")
            self.clock._step_size = step_size
        super().step()
        self.clock._step_size = old_step_size

    def take_steps(self, number_of_steps=1, step_size=None, with_logging=True):
        if not isinstance(number_of_steps, int):
            raise ValueError('Number of steps must be an integer.')

        if run_from_ipython() and with_logging:
            for _ in log_progress(range(number_of_steps), name='Step'):
                self.step(step_size)
        else:
            for _ in range(number_of_steps):
                self.step(step_size)

    def list_values(self):
        return list(self.values.keys())

    def get_values(self):
        return self.values.items()

    def get_value(self, value_pipeline_name):
        return self.values.get_value(value_pipeline_name)

    def list_events(self):
        return self.events.list_events()

    def get_listeners(self, event_name):
        if event_name not in self.events:
            raise ValueError(f'No event {event_name} in system.')
        return self.events.get_listeners(event_name)

    def get_emitter(self, event_name):
        if event_name not in self.events:
            raise ValueError(f'No event {event_name} in system.')
        return self.events.get_emitter(event_name)

    def get_components(self):
        return [component for component in
                self.component_manager._components + self.component_manager._managers + self.component_manager._globals]

    def reload_component(self, component):
        raise NotImplementedError()

    def replace_component(self, old_component, new_component):
        self.component_manager._components.remove(old_component)
        new_component.setup(self.builder)
        self.component_manager.add_components([new_component])

def initialize_simulation(components, input_config=None, plugin_config=None, context_class=InteractiveContext):
    config = build_simulation_configuration()
    config.update(input_config)

    base_plugin_config = ConfigTree(DEFAULT_PLUGINS['plugins'])
    if plugin_config:
        base_plugin_config.update(plugin_config)

    plugin_manager = PluginManager(base_plugin_config)

    simulation = context_class(config, components, plugin_manager)

    return simulation

def setup_simulation(components, input_config=None, plugin_config=None):
    simulation = initialize_simulation(components, input_config, plugin_config)
    simulation.setup()
    simulation.initialize_simulants()

    return simulation


def setup_simulation_from_model_specification(model_specification_file):
    model_specification = build_model_specification(model_specification_file)

    plugin_config = model_specification.plugins
    component_config = model_specification.components
    simulation_config = model_specification.configuration

    plugin_manager = PluginManager(plugin_config)
    component_config_parser = plugin_manager.get_plugin('component_configuration_parser')
    components = component_config_parser.get_components(component_config)

    simulation = InteractiveContext(simulation_config, components, plugin_manager)
    simulation.setup()
    simulation.initialize_simulants()
    return simulation
