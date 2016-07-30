# ~/ceam/ceam/modules/smoking.py

import os.path

import pandas as pd
import numpy as np
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache
from ceam import config
from ceam.gbd_data.gbd_ms_functions import get_exposures

from ceam.engine import SimulationModule


class SmokingModule(SimulationModule):
    """
    Model smoking. Simulants will be smoking at any moment based on whether their `smoking_susceptibility` is less than
    the current smoking prevalence for their demographic.

    NOTE: This does not track whether a simulant has a history of smoking, only what their current state is.

    Population Columns
    ------------------
    smoking_susceptibility
        Likelihood that a simulant will smoke
    """

    def __init__(self):
        super(SmokingModule, self).__init__()
        self.mediation_factor = 0.2

    def setup(self):
        paf_smok = 0.4
        self.incidence_mediation_factors['heart_attack'] = paf_smok * (1 - self.mediation_factor)
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'heart_attack')
        self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'hemorrhagic_stroke')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame(np.random.uniform(low=0.01, high=0.99, size=population_size), columns=['smoking_susceptibility'])

    def load_data(self, path_prefix):
        # TODO: Where does prevalence data come from?
        # The exposure comes from the central comp get draws function (see Everett if there are other questions)
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        lookup_table = load_data_from_cache(get_exposures, 'prevalence', config.getint('simulation_parameters', 'location_id'), year_start, year_end, 166) 

        # STEAL THE TEST BELOW TO PUT INTO THE FUNCTION
        # "Pre-conditions check"

        expected_rows = set((age, sex, year) for age in range(1, 104)
                            for sex in ['Male', 'Female']
                            for year in range(year_start, year_end+1))
        missing_rows = expected_rows.difference(set(tuple(row)
                                                    for row in lookup_table[['age', 'sex', 'year']].values.tolist()))
        missing_rows = [(age, sex, year, 0) for age, sex, year in missing_rows]
        missing_rows = pd.DataFrame(missing_rows, columns=['age', 'sex', 'year', 'prevalence'])
        missing_rows['sex'] = missing_rows.sex.astype('category')
        return lookup_table.append(missing_rows)

    def incidence_rates(self, population, rates):
        smokers = population.smoking_susceptibility < self.lookup_columns(population, ['prevalence'])['prevalence']
        rates *= (2.2**(1 - self.mediation_factor))**smokers
        return rates


# End.
