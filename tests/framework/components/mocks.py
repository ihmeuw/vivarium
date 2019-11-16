
class MockComponentA:
    def __init__(self, *args, name='mock_component_a'):
        self.name = name
        self.args = args
        self.builder_used_for_setup = None

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name


class MockComponentB:
    def __init__(self, *args, name='mock_component_b'):
        self._name = name
        self.args = args
        self.builder_used_for_setup = None
        self._sub_components = []
        if len(self.args) > 1:
            for arg in self.args:
                self._sub_components.append(MockComponentB(arg, name=arg))

    @property
    def name(self):
        return self._name

    @property
    def sub_components(self):
        return self._sub_components

    def setup(self, builder):
        self.builder_used_for_setup = builder
        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, _, metrics):
        if 'test' in metrics:
            metrics['test'] += 1
        else:
            metrics['test'] = 1
        return metrics

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name


class MockGenericComponent:

    configuration_defaults = {
        'component': {
            'key1': 'val',
            'key2': {
                'subkey1': 'val',
                'subkey2': 'val',
            },
            'key3': ['val', 'val', 'val']
        }
    }

    def __init__(self, name):
        self.name = name
        self.configuration_defaults = {self.name: MockGenericComponent.configuration_defaults['component']}
        self.builder_used_for_setup = None

    def setup(self, builder):
        self.builder_used_for_setup = builder
        self.config = builder.configuration[self.name]

    def __eq__(self, other):
        return type(self) == type(other) and self.name == other.name


class NamelessComponent:
    def __init__(self, *args):
        self.args = args


class Listener(MockComponentB):
    def __init__(self, *args, name='test_listener'):
        super().__init__(*args, name=name)
        self.post_setup_called = False
        self.time_step__prepare_called = False
        self.time_step_called = False
        self.time_step__cleanup_called = False
        self.collect_metrics_called = False
        self.simulation_end_called = False

    def setup(self, builder):
        super().setup(builder)
        builder.event.register_listener('post_setup', self.on_post_setup)
        builder.event.register_listener('time_step__prepare', self.on_time_step__prepare)
        builder.event.register_listener('time_step', self.on_time_step)
        builder.event.register_listener('time_step__cleanup', self.on_time_step__cleanup)
        builder.event.register_listener('collect_metrics', self.on_collect_metrics)
        builder.event.register_listener('simulation_end', self.on_simulation_end)

    def on_post_setup(self, _):
        self.post_setup_called = True

    def on_time_step__prepare(self, _):
        self.time_step__prepare_called = True

    def on_time_step(self, _):
        self.time_step_called = True

    def on_time_step__cleanup(self, _):
        self.time_step__cleanup_called = True

    def on_collect_metrics(self, _):
        self.collect_metrics_called = True

    def on_simulation_end(self, _):
        self.simulation_end_called = True
