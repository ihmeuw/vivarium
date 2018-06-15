from . import interactive


class TestingContext(interactive.InteractiveContext):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name, controller in self.plugin_manager.get_optional_controllers().items():
            setattr(self, name, controller)


def initialize_simulation(components, input_config=None, plugin_config=None):
    return interactive.initialize_simulation(components, input_config, plugin_config, TestingContext)


def setup_simulation(components, input_config=None, plugin_config=None):
    simulation = initialize_simulation(components, input_config, plugin_config)
    simulation.setup()
    simulation.initialize_simulants()
    return simulation
