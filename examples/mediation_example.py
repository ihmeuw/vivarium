import pandas as pd
import numpy as np

from ceam.engine import Simulation
from ceam.modules import SimulationModule
from ceam.util import filter_for_rate

class SimpleDisease(SimulationModule):
    def setup(self):
        self.register_value_source(self.incidence_rates, 'incidence_rates', 'simple_disease')
        self.register_event_listener(self.get_sick, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame([False]*population_size, columns=['sick'])

    def incidence_rates(self, population):
        base_rates = pd.Series(0.0001, index=population.index)
        joint_mediated_paf = self.simulation.population_attributable_fraction(population, 'simple_disease')

        return base_rates * (1 - joint_mediated_paf)

    def get_sick(self, event):
        incidence_rate = self.simulation.incidence_rates(event.affected_population, 'simple_disease')
        affected_population = filter_for_rate(event.affected_population, incidence_rate)
        self.simulation.population.loc[affected_population.index, 'sick'] = True


class SimpleRisk(SimulationModule):
    def __init__(self, name, disease, paf, mediation_factor):
        super(SimpleRisk, self).__init__()
        self.name = name
        self.disease_label = disease
        self.paf = paf
        self.mediation_factor = mediation_factor

    def module_id(self):
        return '{} {}'.format(self.__class__, self.name)

    def setup(self):
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', self.disease_label)
        self.register_value_mutator(self.population_attributable_fraction, 'PAF', self.disease_label)

    def load_population_columns(self, path_prefix, population_size):
        # Give everyone a random exposure
        return pd.DataFrame(np.random.random(population_size) + 1, columns=[self.name])

    def load_data(self, path_prefix):
        # NOTE: In the real world I think we would load a table here
        # but for simplicity I'm using a homogeneous PAF for every demographic
        rows = []
        for age in range(104):
            for sex in [1,2]:
                for year in range(1990, 2015):
                    rows.append((age, sex, year, self.paf))
        return pd.DataFrame(rows, columns=['age', 'sex', 'year', self.disease_label+'_PAF'])

    def population_attributable_fraction(self, population, other_paf):
        paf = self.lookup_columns(population, [self.disease_label+'_PAF'])[self.disease_label+'_PAF'].values
        return other_paf * (1 - paf * (1 - self.mediation_factor))

    def incidence_rates(self, population, rates):
        rr = population[self.name] # in this example rr == exposure
        mediated_rr = rr**(1 - self.mediation_factor)
        return rates * population[self.name]

def main():
    simulation = Simulation()
    modules = [
            SimpleDisease(),
            SimpleRisk('risk_one', 'simple_disease', paf=0.9, mediation_factor=1.0),
            SimpleRisk('risk_two', 'simple_disease', paf=0.1, mediation_factor=0.6)
            ]
    for m in modules:
        m.setup()
    simulation.add_children(modules)

    simulation.load_population()
    simulation.load_data()

    start_time = pd.Timestamp('1/1/1990')
    end_time = pd.Timestamp('12/31/2013')
    time_step = pd.Timedelta(days=30.5)

    simulation.run(start_time, end_time, time_step)

    print('Deaths:', (simulation.population.alive == False).sum())
    print('Sick:', (simulation.population.sick == True).sum())

if __name__ == '__main__':
    main()
