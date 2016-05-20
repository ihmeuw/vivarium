from datetime import datetime, timedelta
from collections import defaultdict

from engine import Simulation, SimulationModule
from modules.ihd import IHDModule
from modules.healthcare_access import HealthcareAccessModule

class MetricsModule(SimulationModule):
    def setup(self):
        self.metrics = defaultdict(int)
        self.register_event_listener(self.event_sums, 'healthcare_access')
        self.register_event_listener(self.event_sums, 'deaths')

    def event_sums(self, label, mask, simulation):
        self.metrics[label] += sum(mask)

    def reset(self):
        self.metrics = defaultdict(int)

def main():
    simulation = Simulation()

    for module in [IHDModule(), HealthcareAccessModule()]:
        module.setup()
        simulation.register_module(module)
    metrics_module = MetricsModule()
    metrics_module.setup()
    simulation.register_module(metrics_module)

    simulation.load_population('/home/j/Project/Cost_Effectiveness/dev/data_processed/population_columns')
    simulation.load_data('/home/j/Project/Cost_Effectiveness/dev/data_processed')
    
    for i in range(10):
        simulation.run(datetime(1990, 1, 1), datetime(2013, 12, 31), timedelta(days=30.5)) #TODO: Is 30.5 days a good enough approximation of one month? -Alec
        print('YLDs: %s'%sum(simulation.yld_by_year.values()))
        print(metrics_module.metrics)
        print(sum(simulation.population.alive == True))
        simulation.reset()


if __name__ == '__main__':
    main()
