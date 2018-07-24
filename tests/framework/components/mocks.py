
class MockComponentA:
    def __init__(self, *args):
        self.args = args
        self.builder_used_for_setup = None


class MockComponentB(MockComponentA):
    def setup(self, builder):
        self.builder_used_for_setup = builder

        if len(self.args) > 1:
            children = []
            for arg in self.args:
                children.append(MockComponentB(arg))
            builder.components.add_components(children)
        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, _, metrics):
        if 'test' in metrics:
            metrics['test'] += 1
        else:
            metrics['test'] = 1
        return metrics


class Listener(MockComponentB):
    def __init__(self, *args):
        super().__init__(*args)
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
