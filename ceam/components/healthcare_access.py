# ~/ceam/ceam/modules/healthcare_access.py

from collections import defaultdict
from datetime import timedelta

import numpy as np, pandas as pd

from ceam import config

from ceam.framework.event import listens_for
from ceam.framework.util import from_yearly
from ceam.framework.population import uses_columns
from ceam.framework.values import modifies_value

from ceam_inputs.gbd_ms_functions import load_data_from_cache, get_modelable_entity_draws

# draw random costs for doctor visit (time-specific)
draw = config.getint('run_configuration', 'draw_number')
assert config.getint('simulation_parameters', 'location_id') == 180, 'FIXME: currently cost data for Kenya only'

cost_df = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/doctor_visit_cost_KEN_20160804.csv', index_col=0)
cost_df.index = cost_df.year_id
appointment_cost = cost_df['draw_{}'.format(draw)]


class HealthcareAccess:
    """Model health care utilization. This includes access events due to
    chance (broken arms, flu, etc.) and those due to follow up
    appointments, which are affected by adherence rate. This module
    does not schedule follow-up visits.  But it implements the
    response to follow-ups added to the `healthcare_followup_date`
    column by other modules (for example
    opportunistic_screening.OpportunisticScreeningModule).

    Population Columns
    ------------------
    healthcare_last_visit_date : pd.Timestamp
        most recent health care access event

    healthcare_followup_date : pd.Timestamp
        next scheduled follow-up appointment
    """

    def setup(self, builder):
        draw_number = config.getint('run_configuration', 'draw_number')
        r = np.random.RandomState(123456+draw)
        self.semi_adherent_pr = r.normal(0.4, 0.0485)

        self.cost_by_year = defaultdict(float)
        self.general_access_count = 0
        self.followup_access_count = 0

        self.general_healthcare_access_emitter = builder.emitter('general_healthcare_access')
        self.followup_healthcare_access_emitter = builder.emitter('followup_healthcare_access')

        self.load_utilization(builder)

        self.general_random = builder.randomness('healthcare_general_acess')
        self.followup_random = builder.randomness('healthcare_followup_acess')

    def load_utilization(self, builder):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')
        # me_id 9458 is 'out patient visits'
        # measure 18 is 'Proportion'
        # TODO: Currently this is monthly, not anually
        lookup_table = load_data_from_cache(get_modelable_entity_draws, col_name='utilization_proportion',
                                            year_start=year_start, year_end=year_end, location_id=location_id, measure=18, me_id=9458)
        self.utilization_proportion = builder.lookup(lookup_table)

    @listens_for('initialize_simulants')
    @uses_columns(['healthcare_followup_date', 'healthcare_last_visit_date'])
    def load_population_columns(self, event):
        population_size = len(event.index)
        event.population_view.update(pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size,
                             'healthcare_last_visit_date': [pd.NaT]*population_size}))

    @listens_for('time_step')
    @uses_columns(['healthcare_last_visit_date'], 'alive')
    def general_access(self, event):
        # determine population who accesses care
        t = self.utilization_proportion(event.index)
        index = self.general_random.filter_for_probability(event.index, t)  # FIXME: currently assumes timestep is one month

        # for those who show up, emit_event that the visit has happened, and tally the cost
        event.population_view.update(pd.Series(event.time, index=index))
        self.general_healthcare_access_emitter(event.split(index))
        self.general_access_count += len(index)

        year = event.time.year
        self.cost_by_year[year] += len(index) * appointment_cost[year]

    @listens_for('time_step')
    @uses_columns(['healthcare_last_visit_date', 'healthcare_followup_date', 'adherence_category'], 'alive')
    def followup_access(self, event):
        time_step = timedelta(days=config.getfloat('simulation_parameters', 'time_step'))
        # determine population due for a follow-up appointment
        rows = (event.population.healthcare_followup_date > event.time-time_step) \
               & (event.population.healthcare_followup_date <= event.time)
        affected_population = event.population[rows]

        # of them, determine who shows up for their follow-up appointment
        adherence = pd.Series(1, index=affected_population.index)
        adherence[affected_population.adherence_category == 'non-adherent'] = 0
        semi_adherents = affected_population.loc[affected_population.adherence_category == 'semi-adherent']
        adherence[semi_adherents.index] = self.semi_adherent_pr
        affected_population = self.followup_random.filter_for_probability(affected_population, adherence)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        event.population_view.update(pd.Series(event.time, index=affected_population.index, name='healthcare_last_visit_date'))
        self.followup_healthcare_access_emitter(event.split(affected_population.index))
        self.followup_access_count += len(affected_population)

        year = event.time.year
        self.cost_by_year[year] += len(affected_population) * appointment_cost[year]

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        metrics['healthcare_access_cost'] = sum(self.cost_by_year.values())
        metrics['general_healthcare_access'] = self.general_access_count
        metrics['followup_healthcare_access'] = self.followup_access_count

        if 'cost' in metrics:
            metrics['cost'] += metrics['healthcare_access_cost']
        else:
            metrics['cost'] = metrics['healthcare_access_cost']
        return metrics


# End.
