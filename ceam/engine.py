# ~/ceam/ceam/engine.py

import os
import os.path
from collections import defaultdict
try:
    from configparser import ConfigParser
except ImportError:
    #Python2
    from ConfigParser import SafeConfigParser as ConfigParser

import pandas as pd
import numpy as np
np.seterr(all='raise')
pd.set_option('mode.chained_assignment', 'raise')

from ceam.util import from_yearly, filter_for_rate
from ceam.events import PopulationEvent, only_living
from ceam.modules import ModuleRegistry, SimulationModule


class BaseSimulationModule(SimulationModule):
    def __init__(self):
        super(BaseSimulationModule, self).__init__()
        self.register_event_listener(self.mortality_handler, 'time_step', priority=1)

    def load_population_columns(self, path_prefix, population_size):
        self.population_columns = self.population_columns.join(pd.read_csv(os.path.join(path_prefix, 'age.csv')), how='outer')
        self.population_columns = self.population_columns.assign(fractional_age=self.population_columns.age.astype(float))
        self.population_columns = self.population_columns.join(pd.read_csv(os.path.join(path_prefix, 'sex.csv')))
        self.population_columns = self.population_columns.join(pd.DataFrame({'alive': [True]*len(self.population_columns.age)}))
        self.register_value_source(self.mortality_rates, 'mortality_rates')

    def load_data(self, path_prefix):
        self.lookup_table = pd.read_csv(os.path.join(path_prefix, 'Mortality_Rates.csv'))
        self.lookup_table.columns = [col.lower() for col in self.lookup_table.columns]

    def mortality_rates(self, population):
        return self.lookup_columns(population, ['mortality_rate'])['mortality_rate'].values

    @only_living
    def mortality_handler(self, event):
        mortality_rate = self.simulation.mortality_rates(event.affected_population)
        affected_population = filter_for_rate(event.affected_population, mortality_rate)
        if not affected_population.empty:
            self.simulation.population.loc[affected_population.index, 'alive'] = False
            self.simulation.emit_event(PopulationEvent('deaths', affected_population))

class Simulation(ModuleRegistry):
    def __init__(self, base_module_class=BaseSimulationModule):
        ModuleRegistry.__init__(self, base_module_class)

        self.reference_data = {}
        self.current_time = None
        self.yll_by_year = defaultdict(float)
        self.yld_by_year = defaultdict(float)
        self.deaths_by_year_and_cause = defaultdict(lambda: defaultdict(int))
        self.yll_by_year_and_cause = defaultdict(lambda: defaultdict(float))
        self.new_cases_per_year = defaultdict(lambda: defaultdict(int))
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
            if not module.lookup_table.empty:
                prefixed_table = module.lookup_table.rename(columns=lambda c: column_prefixer(c, module.lookup_column_prefix))
                assert prefixed_table.duplicated(['age','sex','year']).sum() == 0, "%s has a lookup table with duplicate rows"%(module.module_id())

                if lookup_table.empty:
                    lookup_table = prefixed_table
                else:
                    lookup_table = lookup_table.merge(prefixed_table, on=['age', 'sex', 'year'], how='inner')
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
            population = self.population.merge(self.lookup_table[['year','age','sex','lookup_id']], on=['year','age','sex'])
            assert len(population) == len(self.population), "One of the lookup tables is missing rows or has duplicate rows"
            self.population = population

    def incidence_mediation_factor(self, label):
        factor = 1
        for module in self._ordered_modules:
            factor *= 1 - module.incidence_mediation_factors.get(label, 1)
        return 1 - factor


    def emit_event(self, event):
        for module in self._ordered_modules:
            module.emit_event(event)

    def _validate_value_nodes(self):
        sources = defaultdict(lambda: defaultdict(set))
        for module in self._ordered_modules:
            for value_type, msources in module._value_sources.items():
                for label, source in msources.items():
                    sources[value_type][label].add(source)

        duplicates = [(value_type, label, sources) for value_type, by_label in sources.items() for label, sources in by_label.items() if len(sources) > 1]
        assert not duplicates, "Multiple sources for these values: %s"%duplicates

        for module in self._ordered_modules:
            for value_type, mmutators in module._value_mutators.items():
                for label, mutators in mmutators.items():
                    assert sources[value_type][label], "Missing source for mutator: %s"%((value_type, label, mutator))

    def _get_value(self, population, value_type, label=None):
        source = None
        for module in self._ordered_modules:
            if label in module._value_sources[value_type]:
                source = module._value_sources[value_type][label]
                break
        assert source is not None, "No source for %s %s"%(value_type, label)

        mutators = set()
        for module in self._ordered_modules:
            mutators.update(module._value_mutators[value_type][label])

        value = source(population)

        try:
            # If the value has the concept of length, assure that it's length doesn't change over the course of the mutation
            fixed_length = len(value)
        except TypeError:
            fixed_length = None

        for mutator in mutators:
            value = mutator(population, value)
            if fixed_length is not None:
                assert len(value) == fixed_length, "%s is corrupting incidence rates"%mutator
        return value

    def mortality_rates(self, population):
        rates = self._get_value(population, 'mortality_rates')
        return from_yearly(rates, self.last_time_step)

    def incidence_rates(self, population, label):
        rates = self._get_value(population, 'incidence_rates', label)
        return from_yearly(rates, self.last_time_step)

    def disability_weight(self):
        weights = 1
        pop = self.population.loc[self.population.alive == True]
        for module in self._ordered_modules:
            weights *= 1 - module.disability_weight(pop)
        total_weight = 1 - weights
        return total_weight

    def _verify_tables(self, start_time, end_time):
        # Check that all the data necessary to run the requested date range is available
        expected_index = set((age, sex, year) for age in range(1, 104) for sex in [1,2] for year in range(start_time.year, end_time.year))
        for module in self._ordered_modules:
            if not module.lookup_table.empty:
                index = set(tuple(row) for row in module.lookup_table[['age','sex','year']].values.tolist())
                assert len(expected_index.difference(index)) == 0, "%s has a lookup table that doesn't meet the minimal index requirements"%((module.module_id(),))

    def _validate(self, start_time, end_time):
        self._verify_tables(start_time, end_time)
        self._validate_value_nodes()

    def _step(self, time_step):
        self.last_time_step = time_step
        self.population['year'] = self.current_time.year
        self.population.loc[self.population.alive == True, 'fractional_age'] += time_step.days/365.0
        self.population['age'] = self.population.fractional_age.astype(int)
        self.index_population()
        self.emit_event(PopulationEvent('time_step__continuous', self.population))
        self.emit_event(PopulationEvent('time_step', self.population))
        self.current_time += time_step

    def run(self, start_time, end_time, time_step):
        self._validate(start_time, end_time)
        self.reset_population()

        self.current_time = start_time
        while self.current_time <= end_time:
            self._step(time_step)

    def reset(self):
        for module in self._ordered_modules:
            module.reset()
        self.reset_population()
        self.current_time = None




# End.
