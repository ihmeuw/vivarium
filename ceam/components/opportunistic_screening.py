# ~/ceam/ceam/modules/opportunistic_screening.py

from datetime import timedelta
from collections import defaultdict

import numpy as np
import pandas as pd

from ceam import config
from ceam.util import get_draw
from ceam.engine import SimulationModule
from ceam.events import only_living
from ceam.modules.blood_pressure import BloodPressureModule
from ceam.modules.healthcare_access import HealthcareAccessModule
import ceam.modules.healthcare_access

#TODO: This feels like configuration but is difficult to express in ini type files.
MEDICATIONS = [
    {
        'name': 'Thiazide-type diuretics',
        'daily_cost': 0.009,
        'efficacy_mean': 8.8,
        'efficacy_sd': .281,
    },
    {
        'name': 'Beta blockers',
        'daily_cost': 0.048,
        'efficacy_mean': 9.2,
        'efficacy_sd': .332,
    },
    {
        'name': 'ACE Inhibitors',
        'daily_cost': 0.059,
        'efficacy_mean': 10.3,
        'efficacy_sd': .281,
    },
    {
        'name': 'Calcium-channel blockers',
        'daily_cost': 0.166,
        'efficacy_mean': 8.8,
        'efficacy_sd': .23,
    },
]


def _hypertensive_categories(population):
    hypertensive_threshold = config.getint('opportunistic_screening', 'hypertensive_threshold')
    severe_hypertensive_threshold = config.getint('opportunistic_screening', 'severe_hypertensive_threshold')
    under_60 = population.age < 60
    over_60 = population.age >= 60
    under_hypertensive = population.systolic_blood_pressure < hypertensive_threshold
    under_hypertensive_older = population.systolic_blood_pressure < hypertensive_threshold+10
    under_severe_hypertensive = population.systolic_blood_pressure < severe_hypertensive_threshold

    normotensive = under_60 & (under_hypertensive)
    normotensive |= over_60 & (under_hypertensive_older)

    hypertensive = under_60 & (~under_hypertensive) & (under_severe_hypertensive)
    hypertensive |= over_60 & (~under_hypertensive_older) & (under_severe_hypertensive)

    severe_hypertension = (~under_severe_hypertensive)

    return (population.loc[normotensive], population.loc[hypertensive], population.loc[severe_hypertension])


class OpportunisticScreeningModule(SimulationModule):
    """
    Model an intervention where simulants have their blood pressure tested every time they access health care and are prescribed
    blood pressure reducing medication if they are found to be hypertensive. Each simulant can be prescribed up to
    `config.getint('opportunistic_screening', 'max_medications')` drugs. If they are still hypertensive while taking all the drugs then there is no further treatment.

    Population Columns
    ------------------
    medication_count : int
    MEDICATION_supplied_until : pd.Timestamp
    """

    DEPENDENCIES = (BloodPressureModule, HealthcareAccessModule,)
    def __init__(self):
        SimulationModule.__init__(self)
        self.cost_by_year = defaultdict(int)
        self.active = True

        # draw random costs and effects for medications
        draw = config.getint('run_configuration', 'draw_number')
        r = np.random.RandomState(12345+draw)
        cost_df = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/higashi_drug_costs_20160804.csv', index_col='name')

        for med in MEDICATIONS:
            med['efficacy'] = r.normal(loc=med['efficacy_mean'], scale=med['efficacy_sd'])
            med['daily_cost'] = cost_df.loc[med['name'], 'draw_{}'.format(draw)]

        self.semi_adherent_efficacy = r.normal(0.4, 0.0485)

    def setup(self):
         # time_step__continuous happens first, and SBP needs to be
         # updated before everything else which runs during time_step
        self.register_event_listener(self.adjust_blood_pressure, 'time_step__continuous')

        # *_healthcare_access is emitted by HealthcareAccessModule
        self.register_event_listener(self.general_blood_pressure_test, 'general_healthcare_access')
        self.register_event_listener(self.followup_blood_pressure_test, 'followup_healthcare_access')

        assert config.getint('opportunistic_screening', 'max_medications') <= len(MEDICATIONS), 'cannot model more medications than we have data for'

    def load_population_columns(self, path_prefix, population_size):
        #TODO: Some people will start out taking medications?
        population = pd.DataFrame({'medication_count': np.zeros(population_size)})
        for medication in MEDICATIONS:
            population[medication['name']+'_supplied_until'] = pd.NaT
        return population

    def _medication_costs(self, population):
        current_time = pd.Timestamp(self.simulation.current_time)
        for medication_number, medication in enumerate(MEDICATIONS):
            affected_population = population[population.medication_count > medication_number]
            if not affected_population.empty:
                supply_remaining = affected_population[medication['name']+'_supplied_until'] - current_time
                supply_remaining = supply_remaining.fillna(pd.Timedelta(days=0))
                idx = supply_remaining < pd.Timedelta(days=0)
                supply_remaining[idx] = pd.Series([pd.Timedelta(days=0)]*idx.sum())

                supply_needed = self.simulation.population.loc[affected_population.index, 'healthcare_followup_date'] - current_time
                supply_needed = supply_needed.fillna(pd.Timedelta(days=0))
                supply_needed[supply_needed < pd.Timedelta(days=0)] = pd.Timedelta(days=0)

                supplied_until = current_time + pd.DataFrame([supply_needed, supply_remaining]).T.max(axis=1)
                if self.active:
                    self.simulation.population.loc[affected_population.index, medication['name']+'_supplied_until'] = supplied_until
                self.cost_by_year[self.simulation.current_time.year] += max(0, (supply_needed - supply_remaining).dt.days.sum()) * medication['daily_cost']

    def general_blood_pressure_test(self, event):
        #TODO: Model blood pressure testing error

        minimum_age_to_screen = config.getint('opportunistic_screening', 'minimum_age_to_screen')
        affected_population = event.affected_population[event.affected_population.age >= minimum_age_to_screen]

        year = self.simulation.current_time.year
        appointment_cost = ceam.modules.healthcare_access.appointment_cost[year]
        cost_per_simulant = appointment_cost * 0.25  # see CE-94 for discussion
        self.cost_by_year[year] += cost_per_simulant * len(affected_population)

        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(affected_population)

        if self.active:
            # Normotensive simulants get a 60 month followup and no drugs
            self.simulation.population.loc[normotensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5*60)

            # Hypertensive simulants get a 1 month followup and no drugs
            self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5)

            # Severe hypertensive simulants get a 1 month followup and two drugs
            self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = self.simulation.current_time + timedelta(days=30.5*6)

            self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension['medication_count'] + 2, config.getint('opportunistic_screening', 'max_medications'))

        self._medication_costs(affected_population)

    def followup_blood_pressure_test(self, event):
        year = self.simulation.current_time.year
        appointment_cost = ceam.modules.healthcare_access.appointment_cost[year]
        cost_per_simulant = appointment_cost

        self.cost_by_year[year] += cost_per_simulant * len(event.affected_population)
        normotensive, hypertensive, severe_hypertension = _hypertensive_categories(event.affected_population)

        nonmedicated_normotensive = normotensive.loc[normotensive.medication_count == 0]
        medicated_normotensive = normotensive.loc[normotensive.medication_count > 0]

        # Unmedicated normotensive simulants get a 60 month followup
        follow_up = self.simulation.current_time + timedelta(days=30.5*60)
        if self.active:
            self.simulation.population.loc[nonmedicated_normotensive.index, 'healthcare_followup_date'] = follow_up

        # Medicated normotensive simulants get an 11 month followup
        follow_up = self.simulation.current_time + timedelta(days=30.5*11)
        if self.active:
            self.simulation.population.loc[medicated_normotensive.index, 'healthcare_followup_date'] = follow_up

        # Hypertensive simulants get a 6 month followup and go on one drug
        follow_up = self.simulation.current_time + timedelta(days=30.5*6)
        if self.active:
            self.simulation.population.loc[hypertensive.index, 'healthcare_followup_date'] = follow_up
            self.simulation.population.loc[hypertensive.index, 'medication_count'] = np.minimum(hypertensive['medication_count'] + 1, config.getint('opportunistic_screening', 'max_medications'))
            self.simulation.population.loc[severe_hypertension.index, 'healthcare_followup_date'] = follow_up
            self.simulation.population.loc[severe_hypertension.index, 'medication_count'] = np.minimum(severe_hypertension.medication_count + 1, config.getint('opportunistic_screening', 'max_medications'))

        self._medication_costs(event.affected_population)

    @only_living
    def adjust_blood_pressure(self, event):
        for medication_number, medication in enumerate(MEDICATIONS):
            affected_population = event.affected_population[(event.affected_population.medication_count > medication_number) & (event.affected_population[medication['name']+'_supplied_until'] >= self.simulation.current_time - self.simulation.last_time_step)]
            adherence = pd.Series(1, index=affected_population.index)
            adherence[affected_population.adherence_category == 'non-adherent'] = 0
            semi_adherents = affected_population.loc[affected_population.adherence_category == 'semi-adherent']
            adherence[semi_adherents.index] = self.semi_adherent_efficacy

            medication_efficacy = medication['efficacy'] * adherence
            if self.active:
                self.simulation.population.loc[affected_population.index, 'systolic_blood_pressure'] -= medication_efficacy

    def reset(self):
        self.cost_by_year = defaultdict(int)


# End.
