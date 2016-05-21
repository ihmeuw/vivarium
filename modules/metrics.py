from collections import defaultdict

from ceam.engine import SimulationModule

class MetricsModule(SimulationModule):
    def setup(self):
        self.metrics = defaultdict(int)
        self.register_event_listener(self.event_sums, 'healthcare_access')
        self.register_event_listener(self.event_sums, 'deaths')

    def event_sums(self, label, mask, simulation):
        self.metrics[label] += sum(mask)

    def reset(self):
        self.metrics = defaultdict(int)
