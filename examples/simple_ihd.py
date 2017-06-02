import os, time
import numpy as np, pandas as pd
import ceam, ceam.engine, ceam.modules.disease_models, ceam.modules.blood_pressure

class AMIPrevention(ceam.engine.SimulationModule):
    def setup(self):
        self.reset()
        self.register_event_listener(self.track_cost, 'time_step')
        # self.register_value_mutator(self.incidence_rates, 'incidence_rates', 'heart_attack')  # FIXME: does not work

    def track_cost(self, event):
        local_pop = event.affected_population
        # engine='python' should not be necessary, but something seems
        # wrong in pandsa/numexpr right not
        rows = local_pop.eval(self.intervention_group,
                              engine='python')
        self.cumulative_cost += 2.0 * np.sum(rows) * (self.simulation.last_time_step.days / 365.0)

    def incidence_rates(self, population, rates):
        rows = population.eval(self.intervention_group, engine='python')
        rates[rows] *= 0.5
        return rates

    def reset(self):
        self.intervention_group = 'year >= 1995 and age >= 25 and alive == True'
        self.cumulative_cost = 0


class AMIMetrics(ceam.engine.SimulationModule):
    def setup(self):
        self.reset()
        self.register_event_listener(self.count_deaths_and_ylls, 'deaths')
        self.register_event_listener(self.count_ami, 'heart_attacks')  # FIXME: has no effect

    def load_data(self, path_prefix):
        self.life_table = pd.read_csv(os.path.join(path_prefix,
                                                   'interpolated_reference_life_table.csv'))

    def count_deaths_and_ylls(self, event):
        # event.affected_population is a dataframe of individuals
        self.deaths += len(event.affected_population)  
        
        # merge in expected years from life_table (which has columns
        # age and ex)
        t = pd.merge(event.affected_population, self.life_table,
                     on=['age'])
        self.ylls += t.ex.sum()

    def count_ami(self, event):
        self.ami_count += len(event.affected_population)

    def reset(self):
        self.start_time = time.time()
        self.deaths = 0
        self.ylls = 0
        self.ami_count = 0

    def run_time(self):
        return time.time() - self.start_time


### Setup simulation
simulation = ceam.engine.Simulation()
ihd_module = ceam.modules.disease_models.simple_ihd_factory()
ihd_module.setup()  # FIXME: doe not work

metrics_module = AMIMetrics()
metrics_module.setup()

simulation.add_children([ihd_module, metrics_module])

simulation.load_population()
simulation.load_data()


### Run business-as-usual scenario
start_time = pd.Timestamp('1/1/1990')
end_time = pd.Timestamp('12/31/1993')
time_step = pd.Timedelta(days=30.5)                                     # TODO: Is 30.5 days a good enough approximation of one month? -Alec
np.random.seed(123456)                                                  # set random seed for reproducibility
simulation.run(start_time, end_time, time_step)

print('\nWithout intervention:')
print('Deaths:', metrics_module.deaths)
print('YLLs:', metrics_module.ylls)
print('AMIs:', metrics_module.ami_count)
print('Cost:', 0.)
print('Run time:', metrics_module.run_time())


### Run with intervention
intervention = AMIPrevention()
intervention.setup()
simulation.add_children([intervention])

np.random.seed(123456)                                                  # set random seed for reproducibility
simulation.reset()
simulation.run(start_time, end_time, time_step)

print('\nWith intervention:')
print('Deaths:', metrics_module.deaths)
print('YLLs:', metrics_module.ylls)
print('AMIs:', metrics_module.ami_count)
print('Cost:', intervention.cumulative_cost)
print('Run time:', metrics_module.run_time())
