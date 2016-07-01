from datetime import timedelta
from collections import defaultdict

import numpy as np

from ceam import config
from ceam.engine import SimulationModule
from ceam.events import only_living
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.healthcare_access import HealthcareAccessModule

#TODO: This feels like configuration but is difficult to express in ini type files
MEDICATIONS = [
    {
        'name': 'Thiazide-type diuretics',
        'daily_cost': 0.009,
        'efficacy': 8.8,
    },
    {
        'name': 'Calcium-channel blockers',
        'daily_cost': 0.166,
        'efficacy': 8.8,
    },
    {
        'name': 'ACE Inhibitors',
        'daily_cost': 0.059,
        'efficacy': 10.3,
    },
    {
        'name': 'Beta blockers',
        'daily_cost': 0.048,
        'efficacy': 9.2,
    },
]


def _hypertensive_categories(population):
    under_60 = population.age < 60
    over_60 = population.age >= 60
    under_140 = population.systolic_blood_pressure < 140
    under_150 = population.systolic_blood_pressure < 150
    under_180 = population.systolic_blood_pressure < 180

    normotensive = under_60 & (under_140)
    normotensive |= over_60 & (under_150)

    hypertensive = under_60 & (~under_140) & (under_180)
    hypertensive |= over_60 & (~under_150) & (under_180)

    severe_hypertension = (~under_180)

    return (population.loc[normotensive], population.loc[hypertensive], population.loc[severe_hypertension])


class OpportunisticScreeningModule(SimulationModule):
    DEPENDENCIES = (BloodPressureModule, HealthcareAccessModule,)
    def __init__(self):
        SimulationModule.__init__(self)
        self.cost_by_year = defaultdict(int)

    def setup(self):
        self.register_event_listener(self.general_blood_pressure_test, 'general_healthcare_access')
        self.register_event_listener(self.followup_blood_pressure_test, 'followup_healthcare_access')
        self.register_event_listener(self.track_monthly_cost, 'time_step')
        self.register_event_listener(self.adjust_blood_pressure, 'time_step__continuous')

    def load_population_columns(self, path_prefix, population_size):
        #TODO: Some people will start out taking medications?
        self.population_columns['medication_count'] = [0]*population_size

    def general_blood_pressure_test(self, event):
        cost = len(event.affected_population) * config.getfloat('opportunistic_screening', 'blood_pressure_test_cost')
        self.cost_by_year[self.simulation.current_time.year] += cost

        #TODO: testing error

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        # Normotensive simulants get a 60 month followup and no drugs
        self.simulation.population.loc[normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5*60)

        # Hypertensive simulants get a 1 month followup and no drugs
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5)

        # Severe hypertensive simulants get a 1 month followup and two drugs
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5*6)

        self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension['medication_count'] + 2, len(MEDICATIONS))

    def followup_blood_pressure_test(self, event):
        appointment_cost = config.getfloat('appointments', 'cost')
        test_cost = config.getfloat('opportunistic_screening', 'blood_pressure_test_cost')
        cost_per_simulant = appointment_cost + test_cost

        self.cost_by_year[self.simulation.current_time.year] += cost_per_simulant * len(event.affected_population)
        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        nonmedicated_normotensive = normotensive.loc[normotensive.medication_count == 0]
        medicated_normotensive = normotensive.loc[normotensive.medication_count > 0]

        # Unmedicated normotensive simulants get a 60 month followup
        follow_up = self.simulation.current_time + timedelta(days=30.5*60)
        self.simulation.population.loc[nonmedicated_normotensive.index, 'healthcare_followup_date'] = follow_up

        # Medicated normotensive simulants get an 11 month followup
        follow_up = self.simulation.current_time + timedelta(days=30.5*11)
        self.simulation.population.loc[medicated_normotensive.index, 'healthcare_followup_date'] = follow_up

        # Hypertensive simulants get a 6 month followup and go on one drug
        follow_up = self.simulation.current_time + timedelta(days=30.5*6)
        self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = follow_up
        self.simulation.population.loc[hypertensive.index, 'medication_count'] = np.minimum(hypertensive['medication_count'] + 1, len(MEDICATIONS))
        self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = follow_up
        self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension.medication_count + 1, len(MEDICATIONS))


    @only_living
    def track_monthly_cost(self, event):
        for medication_number, medication in enumerate(MEDICATIONS):
            user_count = (event.affected_population.medication_count > medication_number).sum()
            self.cost_by_year[self.simulation.current_time.year] += user_count * medication['daily_cost'] * self.simulation.last_time_step.days

    @only_living
    def adjust_blood_pressure(self, event):
        for medication_number, medication in enumerate(MEDICATIONS):
            medication_efficacy = medication['efficacy'] * config.getfloat('opportunistic_screening', 'adherence')
            affected_population = event.affected_population[event.affected_population.medication_count > medication_number]
            self.simulation.population.loc[affected_population.index, 'systolic_blood_pressure'] -= medication_efficacy

    def reset(self):
        self.cost_by_year = defaultdict(int)
