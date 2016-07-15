# ~/ceam/ceam/engine.py

import os
import os.path
from collections import defaultdict
from functools import reduce

import pandas as pd
import numpy as np
np.seterr(all='raise')
pd.set_option('mode.chained_assignment', 'raise')

from ceam import config
from ceam.tree import Node
from ceam.util import from_yearly, filter_for_rate
from ceam.events import PopulationEvent, Event, only_living
from ceam.modules import ModuleRegistry, SimulationModule, LookupTable, ValueMutationNode, DisabilityWeightMixin

class BaseSimulationModule(SimulationModule):
    def __init__(self):
        super(BaseSimulationModule, self).__init__()
        self.register_event_listener(self.mortality_handler, 'time_step', priority=1)

    def setup(self):
        self.register_value_source(self.mortality_rates, 'mortality_rates')

    def load_population_columns(self, path_prefix, population_size):
        population_columns = pd.read_csv(os.path.join(path_prefix, 'age.csv'))
        population_columns = population_columns.assign(fractional_age=population_columns.age.astype(float))
        population_columns = population_columns.join(pd.read_csv(os.path.join(path_prefix, 'sex.csv')))
        population_columns['alive'] = np.full(len(population_columns), True, dtype=bool)
        population_columns['simulant_id'] = range(0, len(population_columns))
        return population_columns

    def load_data(self, path_prefix):
        lookup_table = pd.read_csv(os.path.join(path_prefix, 'Mortality_Rates.csv'))
        lookup_table.columns = [col.lower() for col in lookup_table.columns]
        return lookup_table

    def mortality_rates(self, population):
        return self.lookup_columns(population, ['mortality_rate'])['mortality_rate'].values

    @only_living
    def mortality_handler(self, event):
        mortality_rate = self.simulation.mortality_rates(event.affected_population)
        affected_population = filter_for_rate(event.affected_population, mortality_rate)
        if not affected_population.empty:
            self.simulation.population.loc[affected_population.index, 'alive'] = False
            self.simulation.emit_event(PopulationEvent('deaths', affected_population))


class Simulation(Node, ModuleRegistry):
    def __init__(self, base_module_class=BaseSimulationModule):
        super(Simulation, self).__init__(base_module_class)

        self.reference_data = {}
        self.current_time = None
        self.yll_by_year = defaultdict(float)
        self.yld_by_year = defaultdict(float)
        self.deaths_by_year_and_cause = defaultdict(lambda: defaultdict(int))
        self.yll_by_year_and_cause = defaultdict(lambda: defaultdict(float))
        self.new_cases_per_year = defaultdict(lambda: defaultdict(int))
        self.population = pd.DataFrame()
        self.initial_population = None
        self.last_time_step = None
        self.lookup_table = LookupTable()

    def load_population(self, path_prefix=None):
        if path_prefix is None:
            path_prefix = config.get('general', 'population_data_directory')


        loaders = []
        for m in self.modules:
            if hasattr(m, 'load_population_columns'):
                loaders.append(m)
            loaders.extend(m.all_decendents(with_attr='load_population_columns'))
        #NOTE: This will always be BaseSimulationModule which loads the core population definition and thus can discover what the population size is
        loader = loaders[0]
        population = [loader.load_population_columns(path_prefix, 0)]
        population_size = len(population[0])

        for loader in loaders[1:]:
            new_pop = loader.load_population_columns(path_prefix, population_size)
            if new_pop is not None:
                assert new_pop.empty or len(new_pop) == population_size, 'Culpret: {0}'.format(loader)
                population.append(new_pop)

        population.append(pd.DataFrame(0, index=np.arange(population_size), columns=['year']))
        self.initial_population = reduce(lambda left,right: left.join(right), population)
        self.reset_population()

    def load_data(self, path_prefix=None):
        self.lookup_table.load_data(self.all_decendents(with_attr='load_data'), path_prefix)

    def lookup_columns(self, population, columns, node):
        return self.lookup_table.lookup_columns(population, columns, node)

    def reset_population(self):
        self.population = self.initial_population.copy()

    def index_population(self):
        if not self.lookup_table.lookup_table.empty:
            if 'lookup_id' in self.population:
                self.population.drop('lookup_id', 1, inplace=True)
            population = self.population.merge(self.lookup_table.lookup_table[['year', 'age', 'sex', 'lookup_id']], on=['year', 'age', 'sex'])
            assert len(population) == len(self.population), "One of the lookup tables is missing rows or has duplicate rows"
            self.population = population

    def incidence_mediation_factor(self, label):
        factor = 1
        for module in self.modules:
            factor *= 1 - module.incidence_mediation_factors.get(label, 1)
        return 1 - factor

    def emit_event(self, event):
        for module in self.modules:
            module.emit_event(event)

    def _validate_value_nodes(self):
        sources = defaultdict(lambda: defaultdict(set))
        for module in self.all_decendents(of_type=ValueMutationNode):
            for value_type, msources in module._value_sources.items():
                for label, source in msources.items():
                    sources[value_type][label].add(source)

        duplicates = [(value_type, label, sources)
                      for value_type, by_label in sources.items()
                      for label, sources in by_label.items()
                      if len(sources) > 1
                     ]
        assert not duplicates, "Multiple sources for these values: %s"%duplicates

        for module in self.all_decendents(of_type=ValueMutationNode):
            for value_type, mmutators in module._value_mutators.items():
                for label, mutators in mmutators.items():
                    assert sources[value_type][label], "Missing source for mutator: {0}. Needed by: {1}".format((value_type, label), mutators)

    def _get_value(self, population, value_type, label=None):
        source = None
        value_nodes = self.all_decendents(of_type=ValueMutationNode)
        for value_node in value_nodes:
            if label in value_node._value_sources[value_type]:
                source = value_node._value_sources[value_type][label]
                break
        assert source is not None, "No source for %s %s"%(value_type, label)

        mutators = set()
        for module in value_nodes:
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
        for node in self.all_decendents(of_type=DisabilityWeightMixin):
            weights *= 1 - node.disability_weight(pop)
        total_weight = 1 - weights
        return total_weight

    def _validate(self, start_time, end_time):
        self._validate_value_nodes()

    def _prepare_step(self, time_step):
        self.population['year'] = self.current_time.year
        self.population.loc[self.population.alive == True, 'fractional_age'] += time_step.days/365.0
        self.population['age'] = self.population.fractional_age.astype(int)
        self.index_population()
        self.last_time_step = time_step

    def _step(self, time_step):
        self._prepare_step(time_step)

        self.emit_event(PopulationEvent('time_step__continuous', self.population))
        self.emit_event(PopulationEvent('time_step', self.population))
        self.emit_event(PopulationEvent('time_step__end', self.population))
        self.current_time += time_step

    def run(self, start_time, end_time, time_step):
        self._validate(start_time, end_time)
        self.reset_population()

        self.current_time = start_time
        self.emit_event(Event('simulation_begin'))
        while self.current_time <= end_time:
            self._step(time_step)
        self.emit_event(Event('simulation_end'))

    def reset(self):
        for module in self.modules:
            module.reset()
        self.reset_population()
        self.current_time = None


# End.
