# ~/ceam/ceam/modules/healthcare_access.py

from collections import defaultdict

import numpy as np, pandas as pd

from ceam import config
from ceam.engine import SimulationModule
from ceam.events import PopulationEvent, only_living
from ceam.util import filter_for_rate, filter_for_probability, from_yearly, get_draw
from ceam.gbd_data.gbd_ms_functions import load_data_from_cache, get_modelable_entity_draws

# draw random costs for doctor visit (time-specific)
draw = config.getint('run_configuration', 'draw_number')
assert config.getint('simulation_parameters', 'location_id') == 180, 'FIXME: currently cost data for Kenya only'

cost_df = pd.read_csv('/home/j/Project/Cost_Effectiveness/dev/data_processed/doctor_visit_cost_KEN_20160804.csv', index_col=0)
cost_df.index = cost_df.year_id
appointment_cost = cost_df['draw_{}'.format(draw)]


class HealthcareAccessModule(SimulationModule):
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

    def __init__(self):
        super(HealthcareAccessModule, self).__init__()
        self.reset()

    def reset(self):
        draw_number = config.getint('run_configuration', 'draw_number')
        r = np.random.RandomState(123456+draw)
        self.semi_adherent_pr = r.normal(0.4, 0.0485)
        self.cost_by_year = defaultdict(float)

    def setup(self):
        self.register_event_listener(self.general_access, 'time_step')
        self.register_event_listener(self.followup_access, 'time_step')

    def load_population_columns(self, path_prefix, population_size):
        return pd.DataFrame({'healthcare_followup_date': [pd.NaT]*population_size,
                             'healthcare_last_visit_date': [pd.NaT]*population_size})

    def load_data(self, path_prefix):
        year_start = config.getint('simulation_parameters', 'year_start')
        year_end = config.getint('simulation_parameters', 'year_end')
        location_id = config.getint('simulation_parameters', 'location_id')
        # me_id 9458 is 'out patient visits'
        # measure 18 is 'Proportion'
        # TODO: Currently this is monthly, not anually
        lookup_table = load_data_from_cache(get_modelable_entity_draws, col_name='utilization_proportion',
                                            year_start=year_start, year_end=year_end, location_id=location_id, measure=18, me_id=9458)
        return lookup_table

    @only_living
    def general_access(self, event):
        # determine population who accesses care
        t = self.lookup_columns(event.affected_population, ['utilization_proportion'])
        affected_population = filter_for_probability(event.affected_population, t['utilization_proportion'])  # FIXME: currently assumes timestep is one month

        # for those who show up, emit_event that the visit has happened, and tally the cost
        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time
        self.simulation.emit_event(PopulationEvent('general_healthcare_access', affected_population))

        year = self.simulation.current_time.year
        self.cost_by_year[year] += len(affected_population) * appointment_cost[year]

    @only_living
    def followup_access(self, event):
        # determine population due for a follow-up appointment
        rows = (event.affected_population.healthcare_followup_date > self.simulation.current_time-self.simulation.last_time_step) \
               & (event.affected_population.healthcare_followup_date <= self.simulation.current_time)
        affected_population = event.affected_population[rows]

        # of them, determine who shows up for their follow-up appointment
        adherence = pd.Series(1, index=affected_population.index)
        adherence[affected_population.adherence_category == 'non-adherent'] = 0
        semi_adherents = affected_population.loc[affected_population.adherence_category == 'semi-adherent']
        adherence[semi_adherents.index] = self.semi_adherent_pr
        affected_population = filter_for_probability(affected_population, adherence)

        # for those who show up, emit_event that the visit has happened, and tally the cost
        self.simulation.population.loc[affected_population.index, 'healthcare_last_visit_date'] = self.simulation.current_time
        self.simulation.emit_event(PopulationEvent('followup_healthcare_access', affected_population))

        year = self.simulation.current_time.year
        self.cost_by_year[year] += len(affected_population) * appointment_cost[year]


# End.
