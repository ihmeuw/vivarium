# ~/ceam/ceam/modules/smoking.py

import os.path

import pandas as pd
import numpy as np
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache
from ceam import config
from ceam.gbd_data.gbd_ms_functions import get_exposures

from ceam.engine import SimulationModule

class SmokingModule(SimulationModule):
    def __init__(self):
        super(SmokingModule, self).__init__()
        self.mediation_factor = 0.2

    def setup(self):
        paf_smok = 0.4
        self.incidence_mediation_factors['ihd'] = paf_smok * (1 - self.mediation_factor)
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'ihd')
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'hemorrhagic_stroke')

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns['smoking_susceptibility'] = np.random.uniform(low=0.01, high=0.99, size=population_size)

    def load_data(self, path_prefix):
        # TODO: Where does prevalence data come from?
	# The exposure comes from the central comp get draws function (see Everett if there are other questions)
        self.lookup_table = load_data_from_cache(get_exposures, 'draw', config.getint('simulation_parameters', 'location_id'), config.getint('simulation_parameters', 'year_start'), config.getint('simulation_parameters', 'year_end'), 166) 
        self.lookup_table.columns = ['age', 'year_id', 'sex_id', 'draw_{i}'.format(i=config.getint('run_configuration', 'draw_number'))]

        # STEAL THE TEST BELOW TO PUT INTO THE FUNCTION
	# "Pre-conditions check"
        expected_rows = set((age, sex, year) for age in range(1, 104)
                            for sex in [1, 2]
                            for year in range(1990, 2011))
        missing_rows = expected_rows.difference(set(tuple(row)
                                                    for row in self.lookup_table[['age', 'sex_id', 'year_id']].values.tolist()))
        missing_rows = [(age, sex, year, 0) for age, sex, year in missing_rows]
        self.lookup_table = self.lookup_table.append(pd.DataFrame(missing_rows, columns=['age', 'sex_id', 'year_id', 'prevalence']))

    def incidence_rates(self, population, rates):
        smokers = population.smoking_susceptibility < self.lookup_columns(population, ['prevalence'])['prevalence']
        rates[smokers] *= 2.2**(1 - self.mediation_factor)
        return rates


# End.
