# ~/ceam/ceam/smoking.py

import os.path

import pandas as pd
import numpy as np

from ceam.engine import SimulationModule
from ceam.modules.ihd import IHDModule
from ceam.modules.hemorrhagic_stroke import HemorrhagicStrokeModule

class SmokingModule(SimulationModule):
    DEPENDENCIES = (IHDModule, HemorrhagicStrokeModule,)
    def setup(self):
        self.mediation_factor = 0.2
        paf_smok = 0.4
        self.incidence_mediation_factors['ihd'] = paf_smok * (1 - self.mediation_factor)

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns['smoking_susceptibility'] = np.random.uniform(low=0.01, high=0.99, size=population_size)

    def load_data(self, path_prefix):
        # TODO: Where does prevalence data come from?
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'smoking_exp_cat1_female.csv'))
        self.lookup_table = self.lookup_table.append(pd.read_csv(os.path.join(path_prefix, 'smoking_exp_cat1_male.csv')))
        self.lookup_table = self.lookup_table.drop_duplicates(['age','year_id','sex_id'])
        self.lookup_table.columns = ['row','age', 'year', 'prevalence', 'sex', 'parameter']
        self.lookup_table = self.lookup_table.drop(['row', 'parameter'], 1)

        missing_rows = set((age, sex, year) for age in range(1, 104) for sex in [1,2] for year in range(1990, 2011)).difference(set(tuple(row) for row in self.lookup_table[['age','sex','year']].values.tolist()))
        missing_rows = [(age,sex,year,0) for age,sex,year in missing_rows]
        self.lookup_table = self.lookup_table.append(pd.DataFrame(missing_rows, columns=['age', 'sex', 'year', 'prevalence']))

    def incidence_rates(self, population, rates, label):
        smokers = population.smoking_susceptibility < self.lookup_columns(population, ['prevalence'])['prevalence']
        if label == 'ihd':
            rates[smokers] *= 2.2**(1 - self.mediation_factor)
        elif label == 'hemorrhagic_stroke':
            rates[smokers] *= 2.2**(1 - self.mediation_factor)
        return rates


# End.
