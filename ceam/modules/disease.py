# ~/ceam/ceam/modules/disease.py

import os.path
from datetime import timedelta
from functools import partial

import pandas as pd
import numpy as np

from ceam import config
from ceam.tree import Node
from ceam.modules import LookupTableMixin, ValueMutationNode, DisabilityWeightMixin
from ceam.events import only_living
from ceam.util import rate_to_probability
from ceam.state_machine import Machine, State, Transition
from ceam.engine import SimulationModule
from ceam.gbd_data import get_excess_mortality, get_incidence
from ceam.gbd_data.gbd_ms_functions import generate_ceam_population, load_data_from_cache, assign_cause_at_beginning_of_simulation



class DiseaseState(State, DisabilityWeightMixin, Node):
    def __init__(self, state_id, disability_weight, dwell_time=0, event_time_column=None, event_count_column=None):
        Node.__init__(self)
        State.__init__(self, state_id)

        self._disability_weight = disability_weight
        self.dwell_time = dwell_time
        if isinstance(self.dwell_time, timedelta):
            self.dwell_time = self.dwell_time.total_seconds()

        if event_time_column:
            self.event_time_column = event_time_column
        else:
            self.event_time_column = self.state_id + '_event_time'

        if event_count_column:
            self.event_count_column = event_count_column
        else:
            self.event_count_column = self.state_id + '_event_count'

    def load_population_columns(self, path_prefix, population_size):
        if self.dwell_time > 0:
            return pd.DataFrame({self.event_time_column: np.zeros(population_size), self.event_count_column: np.zeros(population_size)})

    def next_state(self, population, state_column):
        if self.dwell_time > 0:
            eligible_population = population.loc[population[self.event_time_column] < self.root.current_time.timestamp() - self.dwell_time]
        else:
            eligible_population = population
        return super(DiseaseState, self).next_state(eligible_population, state_column)

    def _transition_side_effect(self, agents, state_column):
        if self.dwell_time > 0:
            agents[self.event_time_column] = self.root.current_time.timestamp()
            agents[self.event_count_column] += 1
        return agents

    def disability_weight(self, population):
        return self._disability_weight * (population[self.parent.condition] == self.state_id)


class ExcessMortalityState(LookupTableMixin, DiseaseState, ValueMutationNode):
    def __init__(self, state_id, modelable_entity_id, **kwargs):
        DiseaseState.__init__(self, state_id, **kwargs)
        ValueMutationNode.__init__(self)

        self.modelable_entity_id = modelable_entity_id

        self.register_value_mutator(self.mortality_rates, 'mortality_rates')

    def load_data(self, prefix_path):
        return get_excess_mortality(self.modelable_entity_id)

    def mortality_rates(self, population, rates):
        return rates + self.lookup_columns(population, ['rate'])['rate'].values * (population[self.parent.condition] == self.state_id)

    def __str__(self):
        return 'ExcessMortalityState("{}", "{}" ...)'.format(self.state_id, self.modelable_entity_id)


class IncidenceRateTransition(LookupTableMixin, Transition, Node, ValueMutationNode):
    def __init__(self, output, rate_label, modelable_entity_id):
        Transition.__init__(self, output, self.probability)
        Node.__init__(self)
        ValueMutationNode.__init__(self)

        self.rate_label = rate_label
        self.modelable_entity_id = modelable_entity_id
        self.register_value_source(self.incidence_rates, 'incidence_rates', rate_label)

    def load_data(self, prefix_path):
        return get_incidence(self.modelable_entity_id)

    def probability(self, agents):
        return rate_to_probability(self.root.incidence_rates(agents, self.rate_label))

    def incidence_rates(self, population):
        mediation_factor = self.root.incidence_mediation_factor(self.parent.condition)
        return pd.Series(self.lookup_columns(population, ['rate'])['rate'].values * mediation_factor, index=population.index)

    def __str__(self):
        return 'IncidenceRateTransition("{0}", "{1}", "{2}")'.format(self.output.state_id, self.rate_label, self.modelable_entity_id)


class DiseaseModule(SimulationModule, Machine):
    def __init__(self, condition):
        SimulationModule.__init__(self)
        Machine.__init__(self, condition)

    def module_id(self):
        return str((self.__class__, self.state_column))

    @property
    def condition(self):
        return self.state_column

    def setup(self):
        self.register_event_listener(self.time_step_handler, 'time_step')

        for state in self.states:
            if isinstance(state, Node):
                self.add_child(state)
            for transition in state.transition_set:
                if isinstance(transition, Node):
                    self.add_child(transition)

    @only_living
    def time_step_handler(self, event):
        # TODO: I shouldn't need this lookup into simulation.population. Somehow this method is
        # poisoning @only_living's cache and this copy prevents that.
        affected_population = self.simulation.population.ix[event.affected_population.index]
        affected_population = self.transition(affected_population)
        self.simulation.population.loc[affected_population.index] = affected_population


    def load_population_columns(self, path_prefix, population_size):
        # TODO: Load real data and integrate with state machine
        state_id_length = max(len(state.state_id) for state in self.states)
        
        location_id = config.getint('simulation_parameters', 'location_id')
        year_start = config.getint('simulation_parameters', 'year_start')

        state_map = {s.state_id:s.modelable_entity_id for s in self.all_decendents(of_type=DiseaseState, with_attr='modelable_entity_id')}

        condition_column = assign_cause_at_beginning_of_simulation(simulants_df=self.simulation.population, location_id=location_id, year_start=year_start, states=state_map)

        population = self.simulation.population.merge(condition_column, on='simulant_id')
        
        population_columns = pd.DataFrame()
        population_columns[self.condition] = population['condition_state'].fillna('healthy')
        return population_columns


# End.
