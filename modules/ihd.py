import os.path
import pandas as pd

from engine import SimulationModule, chronic_condition_incidence_handler

class IHDModule(SimulationModule):
    def setup(self):
        self.register_event_listener(chronic_condition_incidence_handler('ihd'), 'time_step')
        self.track_mortality('ihd')

    def load_population_columns(self, path_prefix):
        self.population_columns = pd.read_csv(os.path.join(path_prefix, 'ihd.csv'))

    def load_data(self, path_prefix):
        self.ihd_mortality_rates = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/ihd_mortality_rate.csv') 
        self.ihd_mortality_rates.columns = [col.lower() for col in self.ihd_mortality_rates]
        self.ihd_incidence_rates = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/IHD incidence rates.csv') 
        self.ihd_incidence_rates.columns = [col.lower() for col in self.ihd_incidence_rates]

    def years_lived_with_disability(self, population):
        return sum(population.ihd == True)*0.08

    def mortality_rates(self, population, rates):
        rates.mortality_rate += population.merge(self.ihd_mortality_rates, on=['age', 'sex', 'year']).mortality_rate
        return rates

    def incidence_rates(self, population, rates, label):
        if label == 'ihd':
            rates.incidence_rate += population.merge(self.ihd_incidence_rates, on=['age', 'sex', 'year']).incidence
            return rates
        return rates
