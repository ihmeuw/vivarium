import time, os

import pandas as pd

from ceam.framework.event import listens_for
from ceam.framework.values import modifies_value
from ceam.framework.population import population_view

class SimpleIntervention:
    intervention_group = 'age >= 25 and alive == True'

    def setup(self, builder):
        self.reset()
        self.year = 1990 # TODO: better plumbing for this information

    @listens_for('time_step')
    @population_view(['age', 'alive'], intervention_group)
    def track_cost(self, event, population_view):
        self.year = event.time.year
        if event.time.year >= 1995:
            local_pop = population_view.get(event.affected_index)
            self.cumulative_cost += 2.0 * len(local_pop) * (event.time_step.days / 365.0) # FIXME: charge full price once per year?

    @modifies_value('mortality_rate')
    @population_view(['age'], intervention_group)
    def mortality_rates(self, index, rates, population_view):
        if self.year >= 1995:
            pop = population_view.get(index)
            rates[pop.index] *= 0.5
        return rates

    def reset(self):
        self.cumulative_cost = 0

    @listens_for('simulation_end', priority=0)
    def dump_metrics(self, event):
        print('Cost:', self.cumulative_cost)

class SimpleMetrics:
    def setup(self, builder):
        self.reset()
        path_prefix = '/home/j/Project/Cost_Effectiveness/dev/data_processed'
        self.life_table = builder.lookup(pd.read_csv(os.path.join(path_prefix, 'interpolated_reference_life_table.csv')), index=('age',))

    @listens_for('deaths')
    def count_deaths_and_ylls(self, event):
        self.deaths += len(event.affected_index)

        t = self.life_table(event.affected_index)
        self.ylls += t.sum()

    def reset(self):
        self.start_time = time.time()
        self.deaths = 0
        self.ylls = 0

    def run_time(self):
        return time.time() - self.start_time

    @listens_for('simulation_end', priority=0)
    def dump_metrics(self, event):
        print('\nWith intervention:')
        print('Deaths:', self.deaths)
        print('YLLs:', self.ylls)
        print('Run time:', self.run_time())
