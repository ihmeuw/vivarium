# ~/ceam/ceam/framework/disease.py

import os.path
from datetime import timedelta
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from .event import listens_for
from .population import uses_columns
from .values import modifies_value, produces_value
from .util import rate_to_probability
from .state_machine import Machine, State, Transition, TransitionSet
import numbers

from ceam_inputs import get_excess_mortality, get_incidence, get_disease_states, get_proportion


class DiseaseState(State):
    def __init__(self, state_id, disability_weight, dwell_time=0, event_time_column=None, event_count_column=None, condition=None):
        State.__init__(self, state_id)

        self.condition = condition
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

    def setup(self, builder):
        columns = [self.condition]
        if self.dwell_time > 0:
            columns += [self.event_time_column, self.event_count_column]
        self.population_view = builder.population_view(columns, 'alive')
        self.clock = builder.clock()

    @listens_for('initialize_simulants')
    def load_population_columns(self, event):
        if self.dwell_time > 0:
            population_size = len(event.index)
            self.population_view.update(pd.DataFrame({self.event_time_column: np.zeros(population_size), self.event_count_column: np.zeros(population_size)}, index=event.index))

    def next_state(self, index, population_view):
        if self.dwell_time > 0:
            population = self.population_view.get(index)
            eligible_index = population.loc[population[self.event_time_column] < self.clock().timestamp() - self.dwell_time].index
        else:
            eligible_index = index
        return super(DiseaseState, self).next_state(eligible_index, population_view)

    def _transition_side_effect(self, index):
        if self.dwell_time > 0:
            pop = self.population_view.get(index)
            pop[self.event_time_column] = self.clock().timestamp()
            pop[self.event_count_column] += 1
            self.population_view.update(pop)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        if self.dwell_time > 0:
            population = self.population_view.get(index)
            metrics[self.event_count_column] = population[self.event_count_column].sum()
        return metrics

    @modifies_value('disability_weight')
    def disability_weight(self, index):
        population = self.population_view.get(index)
        return self._disability_weight * (population[self.condition] == self.state_id)


class ExcessMortalityState(DiseaseState):
    def __init__(self, state_id, modelable_entity_id, prevalence_meid=None, **kwargs):
        DiseaseState.__init__(self, state_id, **kwargs)

        self.modelable_entity_id = modelable_entity_id
        if prevalence_meid:
            # We may be calculating initial prevalence based on a different
            # modelable_entity_id than we use for the mortality rate
            self.prevalence_meid = prevalence_meid
        else:
            self.prevalence_meid = modelable_entity_id

    def setup(self, builder):
        self.mortality = builder.lookup(get_excess_mortality(self.modelable_entity_id))
        return super(ExcessMortalityState, self).setup(builder)

    @modifies_value('mortality_rate')
    def mortality_rates(self, index, rates):
        population = self.population_view.get(index)
        return rates + self.mortality(population.index) * (population[self.condition] == self.state_id)

    @modifies_value('modelable_entity_ids.mortality')
    def mmeids(self):
        return self.modelable_entity_id

    def name(self):
        return '{} ({}, {})'.format(self.state_id, self.modelable_entity_id, self.prevalence_meid)

    def __str__(self):
        return 'ExcessMortalityState("{}", "{}" ...)'.format(self.state_id, self.modelable_entity_id)


class IncidenceRateTransition(Transition):
    def __init__(self, output, rate_label, modelable_entity_id):
        Transition.__init__(self, output, self.probability)

        self.rate_label = rate_label
        self.modelable_entity_id = modelable_entity_id

    def setup(self, builder):
        self.incidence_rates = produces_value('incidence_rate.{}'.format(self.rate_label))(self.incidence_rates)
        self.effective_incidence = builder.rate('incidence_rate.{}'.format(self.rate_label))
        self.effective_incidence.source = self.incidence_rates
        self.joint_paf = builder.value('paf.{}'.format(self.rate_label))
        self.base_incidence = builder.lookup(get_incidence(self.modelable_entity_id))

    def probability(self, index):
        return rate_to_probability(self.effective_incidence(index))

    def incidence_rates(self, index):
        base_rates = self.base_incidence(index)
        joint_mediated_paf = self.joint_paf(index)

        return pd.Series(base_rates.values * joint_mediated_paf.values, index=index)

    def __str__(self):
        return 'IncidenceRateTransition("{0}", "{1}", "{2}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.rate_label, self.modelable_entity_id)


class ProportionTransition(Transition):
    def __init__(self, output, modelable_entity_id=None, proportion=None):
        Transition.__init__(self, output, self.probability)

        if modelable_entity_id and proportion:
            raise ValueError("Must supply modelable_entity_id or proportion (proportion can be an int or df) but not both")
      
        # @alecwd: had to change line below since it was erroring out when proportion is a dataframe. might be a cleaner way to do this that I don't know of
        if modelable_entity_id is None and proportion is None:
           raise ValueError("Must supply either modelable_entity_id or proportion (proportion can be int or df)")

        self.modelable_entity_id = modelable_entity_id
        self.proportion = proportion

    def setup(self, builder):
        if self.modelable_entity_id:
            self.proportion = builder.lookup(get_proportion(self.modelable_entity_id))
        elif not isinstance(self.proportion, numbers.Number):
            self.proportion = builder.lookup(self.proportion)

    def probability(self, index):
        if callable(self.proportion):
            return self.proportion(index)
        else:
            return pd.Series(self.proportion, index=index)

    def label(self):
        if self.modelable_entity_id:
            return str(self.modelable_entity_id)
        else:
            return str(self.proportion)

    def __str__(self):
        return 'ProportionTransition("{}", "{}", "{}")'.format(self.output.state_id if hasattr(self.output, 'state_id') else [str(x) for x in self.output], self.modelable_entity_id, self.proportion)


class DiseaseModel(Machine):
    def __init__(self, condition):
        Machine.__init__(self, condition)

    def module_id(self):
        return str((self.__class__, self.state_column))

    @property
    def condition(self):
        return self.state_column

    def setup(self, builder):
        self.population_view = builder.population_view([self.condition], 'alive')

        sub_components = set()
        for state in self.states:
            state.condition = self.condition
            sub_components.add(state)
            sub_components.add(state.transition_set)
            for transition in state.transition_set:
                sub_components.add(transition)
                if isinstance(transition.output, TransitionSet):
                    sub_components.add(transition.output)
        return sub_components

    @listens_for('time_step')
    def time_step_handler(self, event):
        self.transition(event.index)


    @listens_for('initialize_simulants')
    @uses_columns(['age', 'sex'])
    def load_population_columns(self, event):
        population = event.population

        state_map = {s.state_id:s.prevalence_meid for s in self.states if hasattr(s, 'prevalence_meid')}

        population['sex_id'] = population.sex.apply({'Male':1, 'Female':2}.get)
        condition_column = get_disease_states(population, state_map)
        condition_column = condition_column.rename(columns={'condition_state': self.condition})

        self.population_view.update(condition_column)

    @modifies_value('metrics')
    def metrics(self, index, metrics):
        population = self.population_view.get(index)
        metrics[self.condition + '_count'] = (population[self.condition] != 'healthy').sum()
        return metrics
# End.
