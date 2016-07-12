import os.path
from datetime import timedelta
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.tree import Node
from ceam.modules import DataLoaderMixin, ValueMutationNode
from ceam.util import rate_to_probability
from ceam.state_machine import Machine, State, Transition
from ceam.engine import SimulationModule

class DiseaseState(State, Node):
    def __init__(self, state_id, disability_weight, dwell_time=0, event_time_column=None, event_count_column=None):
        Node.__init__(self)
        State.__init__(self, state_id)

        self.disability_weight = disability_weight
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

    def next_state(self, population, state_column):
        if self.dwell_time > 0:
            eligible_population = population.loc[population[self.event_time_column] >= self.root.current_time.timestamp()]
        else:
            eligible_population = population
        return super(DiseaseState, self).next_state(eligible_population, state_column)

    def _transition_side_effect(self, agents, state_column):
        if self.dwell_time > 0:
            agents[self.event_time_column] = self.root.current_time.timestamp()
            agents[self.event_count_column] += 1
        return agents


class ExcessMortalityState(DiseaseState, DataLoaderMixin, ValueMutationNode):
    def __init__(self, state_id, excess_mortality_table, **kwargs):
        DiseaseState.__init__(self, state_id, **kwargs)
        DataLoaderMixin.__init__(self)
        ValueMutationNode.__init__(self)

        self.excess_mortality_table = excess_mortality_table

        self.register_value_mutator(self.mortality_rates, 'mortality_rates')

    def _load_data(self, prefix_path):
        lookup_table = pd.read_csv(os.path.join(prefix_path, self.excess_mortality_table))
        lookup_table.columns = _rename_rate_column(lookup_table, 'rate')
        lookup_table.drop_duplicates(['age', 'year', 'sex'], inplace=True)
        return lookup_table

    def mortality_rates(self, population, rates):
        return rates + self.lookup_columns(population, ['rate'])['rate'].values * (population[self.parent.condition] == self.state_id)

    def __str__(self):
        return 'ExcessMortalityState("{0}" ...)'.format(self.state_id, self.excess_mortality_table)

class IncidenceRateTransition(Transition, Node, DataLoaderMixin, ValueMutationNode):
    def __init__(self, output, rate_label, incidence_rate_table):
        Transition.__init__(self, output, self.probability)
        Node.__init__(self)
        DataLoaderMixin.__init__(self)
        ValueMutationNode.__init__(self)

        self.rate_label = rate_label
        self.incidence_rate_table = incidence_rate_table
        self.register_value_source(self.incidence_rates, 'incidence_rates', rate_label)

    def _load_data(self, prefix_path):
        lookup_table = pd.read_csv(os.path.join(prefix_path, self.incidence_rate_table))
        lookup_table.columns = _rename_rate_column(lookup_table, 'rate')
        lookup_table.drop_duplicates(['age', 'year', 'sex'], inplace=True)
        return lookup_table

    def probability(self, agents):
        return rate_to_probability(self.root.incidence_rates(agents, self.rate_label))

    def incidence_rates(self, population):
        mediation_factor = self.root.incidence_mediation_factor(self.parent.condition)
        return pd.Series(self.lookup_columns(population, ['rate'])['rate'].values * mediation_factor, index=population.index)

    def __str__(self):
        return 'IncidenceRateTransition("{0}", "{1}", "{2}")'.format(self.output.state_id, self.rate_label, self.incidence_rate_table)


def fancy_heart_disease_factory():
    module = DiseaseModule('ihd')

    healthy = DiseaseState('healthy', disability_weight=0)
    # TODO: disability weight for heart attack
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=0.439, dwell_time=timedelta(days=28), excess_mortality_table='mi_acute_excess_mortality.csv')

    mild_heart_failure = ExcessMortalityState('mild_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    moderate_heart_failure = ExcessMortalityState('moderate_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    severe_heart_failure = ExcessMortalityState('severe_heart_failure', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')
    angina = ExcessMortalityState('angina', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', 'ihd_incidence_rates.csv')
    angina_transition = IncidenceRateTransition(angina, 'angina', 'ihd_incidence_rates.csv')
    healthy.transition_set.add(heart_attack_transition)
    healthy.transition_set.add(angina_transition)

    heart_attack.transition_set.add(Transition(mild_heart_failure))
    heart_attack.transition_set.add(Transition(moderate_heart_failure))
    heart_attack.transition_set.add(Transition(severe_heart_failure))
    heart_attack.transition_set.add(Transition(angina))

    mild_heart_failure.transition_set.add(heart_attack_transition)
    moderate_heart_failure.transition_set.add(heart_attack_transition)
    severe_heart_failure.transition_set.add(heart_attack_transition)
    angina.transition_set.add(heart_attack_transition)

    module.states.update([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, angina])
    return module

def ihd_factory():
    module = DiseaseModule('ihd')

    healthy = DiseaseState('healthy', disability_weight=0)
    # TODO: disability weight for heart attack
    heart_attack = ExcessMortalityState('heart_attack', disability_weight=0.439, dwell_time=timedelta(days=28), excess_mortality_table='mi_acute_excess_mortality.csv')
    chronic_ihd = ExcessMortalityState('chronic_ihd', disability_weight=0.08, excess_mortality_table='ihd_mortality_rate.csv')

    heart_attack_transition = IncidenceRateTransition(heart_attack, 'heart_attack', 'ihd_incidence_rates.csv')
    healthy.transition_set.add(heart_attack_transition)

    heart_attack.transition_set.add(Transition(chronic_ihd))

    chronic_ihd.transition_set.add(heart_attack_transition)

    module.states.update([healthy, heart_attack, chronic_ihd])

    return module

def hemorrhagic_stroke_factory():
    module = DiseaseModule('hemorrhagic_stroke')

    healthy = DiseaseState('healthy', disability_weight=0)
    # TODO: disability weight for stroke
    stroke = ExcessMortalityState('hemorrhagic_stroke', disability_weight=0.92, dwell_time=timedelta(days=28), excess_mortality_table='acute_hem_stroke_excess_mortality.csv')
    chronic_stroke = ExcessMortalityState('chronic_stroke', disability_weight=0.31, excess_mortality_table='chronic_hem_stroke_excess_mortality.csv')

    stroke_transition = IncidenceRateTransition(stroke, 'hemorrhagic_stroke', 'hem_stroke_incidence_rates.csv')
    healthy.transition_set.add(stroke_transition)

    stroke.transition_set.add(Transition(chronic_stroke))

    chronic_stroke.transition_set.add(stroke_transition)

    module.states.update([healthy, stroke, chronic_stroke])

    return module


def _rename_rate_column(table, col_name):
    columns = []
    for col in table.columns:
        col = col.lower()
        if col in ['age', 'sex', 'year']:
            columns.append(col)
        else:
            columns.append(col_name)
    return columns

class DiseaseModule(SimulationModule, Machine):
    def __init__(self, condition):
        SimulationModule.__init__(self)
        Machine.__init__(self, condition)

    def module_id(self):
        return (self.__class__, self.state_column)

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

    def disability_weight(self, population):
        weights = 1
        for state in self.states:
            weights *= 1 - ((population[self.condition] == state.state_id).sum() * state.disability_weight)
        return 1 - weights

    def time_step_handler(self, event):
        affected_population = self.transition(event.affected_population)
        self.simulation.population.loc[affected_population.index] = affected_population


    def load_population_columns(self, path_prefix, population_size):
        # TODO: Load real data and integrate with state machine
        self.population_columns = pd.DataFrame(['healthy']*population_size, columns=[self.condition])
        for state in self.states:
            if state.dwell_time > 0:
                self.population_columns[state.state_id + '_event_count'] = 0
                self.population_columns[state.state_id + '_event_time'] = np.array([0] * population_size, dtype=np.float)
