"""A simple component to allow users to store outputs in the simulation."""


class Metrics:
    """This class declares a value pipeline that allows other components to store summary metrics."""
    def setup(self, builder):
        self.metrics = builder.value.register_value_producer('metrics', source=lambda index: {})
