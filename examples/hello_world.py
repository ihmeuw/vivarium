# ~/ceam/examples/hello_world.py

import os, time
import numpy as np, pandas as pd
import ceam, ceam.engine


class SimpleIntervention(ceam.engine.SimulationModule):
    def setup(self):
        self.reset()
        self.register_event_listener(self.track_cost, 'time_step')
        self.register_value_mutator(self.mortality_rates, 'mortality_rates')

    def track_cost(self, event):
        local_pop = event.affected_population
        rows = local_pop.eval(self.intervention_group, engine='python') # engine='python' should not be necessary, but something seems wrong in pandsa/numexpr right now.
        self.cumulative_cost += 2.0 * np.sum(rows) * (self.simulation.last_time_step.days / 365.0) # FIXME: charge full price once per year?

    def mortality_rates(self, population, rates):
        rows = population.eval(self.intervention_group, engine='python')
        rates[np.where(rows)] *= 0.5
        return rates

    def reset(self):
        self.intervention_group = 'year >= 1995 and age >= 25 and alive == True'
        self.cumulative_cost = 0


class SimpleMetrics(ceam.engine.SimulationModule):
    def setup(self):
        self.reset()
        self.register_event_listener(self.count_deaths_and_ylls, 'deaths')

    def load_data(self, path_prefix):
        self.life_table = pd.read_csv(os.path.join(path_prefix, 'interpolated_reference_life_table.csv'))

    def count_deaths_and_ylls(self, event):
        self.deaths += len(event.affected_population)                   # event.affected_population is a dataframe of individuals

        t = pd.merge(event.affected_population, self.life_table, on=['age']) # merge in expected years from life_table (which has columns age and ex)
        self.ylls += t.ex.sum()

    def reset(self):
        self.start_time = time.time()
        self.deaths = 0
        self.ylls = 0

    def run_time(self):
        return time.time() - self.start_time


### Setup simulation
simulation = ceam.engine.Simulation()

metrics_module = SimpleMetrics()
metrics_module.setup()

simulation.add_children([metrics_module])

simulation.load_population()
simulation.load_data()


### Run business-as-usual scenario
start_time = pd.Timestamp('1/1/1990')
end_time = pd.Timestamp('12/31/2010')
time_step = pd.Timedelta(days=30.5)                                     # TODO: Is 30.5 days a good enough approximation of one month? -Alec
np.random.seed(123456)                                                  # set random seed for reproducibility
simulation.run(start_time, end_time, time_step)

print('\nWithout intervention:')
print('Deaths:', metrics_module.deaths)
print('YLLs:', metrics_module.ylls)
print('Cost:', 0.)
print('Run time:', metrics_module.run_time())


### Run with intervention
intervention = SimpleIntervention()
intervention.setup()
simulation.add_children([intervention])

np.random.seed(123456)                                                  # set random seed for reproducibility
simulation.reset()
simulation.run(start_time, end_time, time_step)

print('\nWith intervention:')
print('Deaths:', metrics_module.deaths)
print('YLLs:', metrics_module.ylls)
print('Cost:', intervention.cumulative_cost)
print('Run time:', metrics_module.run_time())


# End.
