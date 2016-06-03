# ~/ceam/ceam/engine.py

import os
import os.path
from collections import defaultdict, Iterable
try:
    from configparser import ConfigParser
except ImportError:
    #Python2
    from ConfigParser import SafeConfigParser as ConfigParser

import pandas as pd
import numpy as np

from ceam.util import sort_modules, from_yearly_rate, filter_for_rate
from ceam.events import EventHandler, PopulationEvent, only_living



class Simulation(object):
    def __init__(self):
        self.reference_data = {}
        self._modules = {}
        self._ordered_modules = []
        self.current_time = None
        self.yll_by_year = defaultdict(float)
        self.yld_by_year = defaultdict(float)
        self.deaths_by_year_and_cause = defaultdict(lambda: defaultdict(int))
        self.yll_by_year_and_cause = defaultdict(lambda: defaultdict(float))
        self.new_cases_per_year = defaultdict(lambda: defaultdict(int))
        self.register_modules([BaseSimulationModule()])
        self.population = pd.DataFrame()
        self.lookup_table = pd.DataFrame()
        self.config = ConfigParser()

        config_path = os.path.abspath(os.path.dirname(__file__))
        self.config.read([os.path.join(config_path, 'config.cfg'), os.path.join(config_path, 'local.cfg'), os.path.expanduser('~/ceam.cfg')])

    def load_data(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = self.config.get('general', 'reference_data_directory')

        for module in self._ordered_modules:
            module.load_data(path_prefix)
        lookup_table = pd.DataFrame()

        #TODO: This is ugly. There must be a better way
        def column_prefixer(column, prefix):
            if column not in ['age', 'year', 'sex']:
                return prefix + '_' + column
            return column
        for module in self._ordered_modules:
            # TODO: This should guard against badly formatted tables somehow
            if not module.lookup_table.empty:
                prefixed_table = module.lookup_table.rename(columns=lambda c: column_prefixer(c, module.lookup_column_prefix))
                if lookup_table.empty:
                    lookup_table = prefixed_table
                else:
                    lookup_table = lookup_table.merge(prefixed_table, how='outer')
        lookup_table['lookup_id'] = range(0, len(lookup_table))
        self.lookup_table = lookup_table

    def load_population(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = self.config.get('general', 'population_data_directory')

        #NOTE: This will always be BaseSimulationModule which loads the core population definition and thus can discover what the population size is
        module = self._ordered_modules[0]
        module.load_population_columns(path_prefix, 0)
        population_size = len(module.population_columns)

        for module in self._ordered_modules[1:]:
            module.load_population_columns(path_prefix, population_size)
            assert module.population_columns.empty or len(module.population_columns) == population_size, 'Culpret: %s'%module
        self.reset_population()

    def reset_population(self):
        population = pd.DataFrame()
        for module in self._ordered_modules:
            population = population.join(module.population_columns, how='outer')
        self.population = population.join(pd.DataFrame(0, index=np.arange(len(population)), columns=['year']))

    def index_population(self):
        if not self.lookup_table.empty:
            if 'lookup_id' in self.population:
                self.population.drop('lookup_id', 1, inplace=True)
            self.population = self.population.merge(self.lookup_table[['year','age','sex','lookup_id']], on=['year','age','sex'])

    def register_modules(self, modules):
        for module in modules:
            module.register(self)
            self._modules[module.__class__] = module

        # TODO: This little dance is awkward but it makes it so I can privilege BaseSimulationModule without having to import it in utils.
        # It should also probably be happening at a lifecycle phase between here and the loading of data, but that doesn't exist yet.
        to_sort = set(self._modules.values())
        to_sort.remove(self._modules[BaseSimulationModule])
        self._ordered_modules = sort_modules(to_sort, self._modules)
        self._ordered_modules.insert(0, self._modules[BaseSimulationModule])

    def deregister_modules(self, modules):
        for module in modules:
            module.deregister(self)
            del self._modules[module.__class__]

        to_sort = set(self._modules.values())
        to_sort.remove(self._modules[BaseSimulationModule])
        self._ordered_modules = sort_modules(to_sort, self._modules)
        self._ordered_modules.insert(0, self._modules[BaseSimulationModule])

    def emit_event(self, event):
        for module in self._ordered_modules:
            module.emit_event(event)

    def mortality_rates(self, population):
        rates = 0
        for module in self._ordered_modules:
            rates = module.mortality_rates(population, rates)
        return from_yearly_rate(rates, self.last_time_step)

    def incidence_rates(self, population, label):
        rates = 0
        for module in self._ordered_modules:
            rates = module.incidence_rates(population, rates, label)
        return from_yearly_rate(rates, self.last_time_step)

    def disability_weight(self):
        weights = 1
        pop = self.population.loc[self.population.alive == True]
        for module in self._ordered_modules:
            weights *= 1 - module.disability_weight(pop)
        total_weight = 1 - weights
        return total_weight

    def run(self, start_time, end_time, time_step):
        self.reset_population()
        self.current_time = start_time
        self.last_time_step = time_step
        while self.current_time <= end_time:
            self.population['year'] = self.current_time.year
            self.population.loc[self.population.alive == True, 'fractional_age'] += time_step.days/365.0
            self.population['age'] = self.population.fractional_age.astype(int)
            self.index_population()
            self.emit_event(PopulationEvent('time_step', self.population))
            self.current_time += time_step

    def reset(self):
        for module in self._ordered_modules:
            module.reset()
        self.reset_population()
        self.current_time = None


class SimulationModule(EventHandler):
    DEPENDENCIES = set()
    def __init__(self):
        EventHandler.__init__(self)
        self.population_columns = pd.DataFrame()
        self.lookup_table = pd.DataFrame()

    def setup(self):
        pass

    def reset(self):
        pass

    def register(self, simulation):
        self.simulation = simulation

    def deregister(self, simulation):
        pass

    def load_population_columns(self, path_prefix, population_size):
        pass

    def load_data(self, path_prefix):
        pass

    def disability_weight(self, population):
        return 0.0

    def mortality_rates(self, population, rates):
        return rates

    def incidence_rates(self, population, rates, label):
        return rates

    @property
    def lookup_column_prefix(self):
        return self.__class__.__name__

    def lookup_columns(self, population, columns):
        origonal_columns = columns
        columns = [self.lookup_column_prefix + '_' + c for c in columns]
        results = self.simulation.lookup_table.ix[population.lookup_id, columns]
        return results.rename(columns=dict(zip(columns,origonal_columns)))



class BaseSimulationModule(SimulationModule):
    def __init__(self):
        super(BaseSimulationModule, self).__init__()
        self.register_event_listener(self.mortality_handler, 'time_step', priority=1)

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = self.population_columns.join(pd.read_csv(os.path.join(path_prefix, 'age.csv')), how='outer')
        self.population_columns = self.population_columns.assign(fractional_age=self.population_columns.age.astype(float))
        self.population_columns = self.population_columns.join(pd.read_csv(os.path.join(path_prefix, 'sex.csv')))
        self.population_columns = self.population_columns.join(pd.DataFrame({'alive': [True]*len(self.population_columns.age)}))

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'Mortality_Rates.csv'))
        self.lookup_table.columns = [col.lower() for col in self.lookup_table.columns]

    def mortality_rates(self, population, rates):
        return rates + self.lookup_columns(population, ['mortality_rate'])['mortality_rate']

    @only_living
    def mortality_handler(self, event):
        mortality_rate = self.simulation.mortality_rates(event.affected_population)
        affected_population = filter_for_rate(event.affected_population, mortality_rate)
        
        if not affected_population.empty:
            self.simulation.population.loc[affected_population.index, 'alive'] = False
            self.simulation.emit_event(PopulationEvent('deaths', affected_population))


# End.
