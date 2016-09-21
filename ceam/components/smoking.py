# ~/ceam/ceam/modules/smoking.py

import os.path
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value
from ceam.framework.population import uses_columns

from ceam.gbd_data import get_pafs, get_relative_risks, get_exposures


class Smoking:
    """
    Model smoking. Simulants will be smoking at any moment based on whether their `smoking_susceptibility` is less than
    the current smoking prevalence for their demographic.

    NOTE: This does not track whether a simulant has a history of smoking, only what their current state is.

    Population Columns
    ------------------
    smoking_susceptibility
        Likelihood that a simulant will smoke
    """

    def setup(self, builder):

        self.load_prevelence(builder)

        self.load_reletive_risks(builder)

        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ihd_rr), 'incidence_rate.heart_attack')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.hemorrhagic_stroke_rr), 'incidence_rate.hemorrhagic_stroke')
        builder.modifies_value(partial(self.incidence_rates, rr_lookup=self.ischemic_stroke_rr), 'incidence_rate.ischemic_stroke')

        self.load_pafs(builder)

        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ihd_paf), 'paf.heart_attack')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.hemorrhagic_stroke_paf), 'paf.hemorrhagic_stroke')
        builder.modifies_value(partial(self.population_attributable_fraction, paf_lookup=self.ischemic_stroke_paf), 'paf.ischemic_stroke')

        self.randomness = builder.randomness('smoking')

    @listens_for('initialize_simulants')
    @uses_columns(['smoking_susceptibility'])
    def load_susceptibility(self, event):
        event.population_view.update(pd.Series(self.randomness.get_draw(event.index)*0.98+0.01, name='smoking_susceptibility'))

    def load_prevelence(self, builder):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')

        self.exposure = builder.lookup(get_exposures(risk_id=166))

    def load_reletive_risks(self, builder):
        self.ihd_rr = builder.lookup(get_relative_risks(risk_id=166, cause_id=493))
        self.hemorrhagic_stroke_rr = builder.lookup(get_relative_risks(risk_id=166, cause_id=496))
        self.ischemic_stroke_rr = builder.lookup(get_relative_risks(risk_id=166, cause_id=495))

    def load_pafs(self, builder):
        self.ihd_paf = builder.lookup(get_pafs(risk_id=166, cause_id=493))
        self.hemorrhagic_stroke_paf = builder.lookup(get_pafs(risk_id=166, cause_id=496))
        self.ischemic_stroke_paf = builder.lookup(get_pafs(risk_id=166, cause_id=495))

    def population_attributable_fraction(self, index, paf_lookup):
        paf = paf_lookup(index)
        return paf

    @uses_columns(['smoking_susceptibility'])
    def incidence_rates(self, index, rates, population_view, rr_lookup):
        population = population_view.get(index)
        rr = rr_lookup(index)

        smokers = population.smoking_susceptibility < self.exposure(index)
        rates *= rr.values**smokers.values
        return rates


# End.
