import os.path
from datetime import timedelta
from functools import partial

import pandas as pd
import numpy as np

from ceam import config

from ceam.util import rate_to_probability
from ceam.state_machine import Machine, State, Transition
from ceam.engine import SimulationModule

def _probability_for_condition(module, label):
    def inner(agents):
        return rate_to_probability(module.simulation.incidence_rates(agents, label))
    return inner

def _delayed_transition(module, label, time_column, delay):
    prob_func = _probability_for_condition(module, label)
    def inner(agents):
        probs = prob_func(agents)
        return probs.where(agents[time_column] + delay.total_seconds() <= module.simulation.current_time.timestamp(), 0.0)
    return inner

def _build_table(rate, column='rate'):
    start_year = config.getint('simulation_parameters', 'year_start')
    end_year = config.getint('simulation_parameters', 'year_end') + 1
    index = set((age, sex, year) for age in range(1, 104) for sex in [1,2] for year in range(start_year, end_year))
    rows = [(age, sex, year, rate) for age, sex, year in index]
    return pd.DataFrame(rows, columns=['age', 'sex', 'year', column])

def fancy_heart_disease_factory():
    module = DiseaseModule(
                           incidence_rates = {
                               'heart_attack': 'ihd_incidence_rates.csv',
                               'angina': 'ihd_incidence_rates.csv',
                               'mild_heart_failure': _build_table(0.25),
                               'moderate_heart_failure': _build_table(0.25),
                               'severe_heart_failure': _build_table(0.25),
                               'post_heart_attack_angina': _build_table(0.25),
                               'additional_heart_attack': 'ihd_incidence_rates.csv',
                            },
                            mortality_rates = {
                                'healthy': _build_table(0.0),
                                'heart_attack': 'mi_acute_excess_mortality.csv',
                                'mild_heart_failure': 'ihd_mortality_rate.csv',
                                'moderate_heart_failure': 'ihd_mortality_rate.csv',
                                'severe_heart_failure': 'ihd_mortality_rate.csv',
                                'angina': 'ihd_mortality_rate.csv',

                            }
            )
    machine = Machine('ihd')

    healthy = State('healthy')
    def heart_attack_side_effect(agents, condition):
        agents[condition+'_event_time'] = module.simulation.current_time.timestamp()
        agents[condition+'_event_count'] += 1
        return agents
    heart_attack = State('heart_attack', heart_attack_side_effect)
    mild_heart_failure = State('mild_heart_failure')
    moderate_heart_failure = State('moderate_heart_failure')
    severe_heart_failure = State('severe_heart_failure')
    angina = State('angina')

    healthy.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'heart_attack')))
    healthy.transition_set.add(Transition(angina, _probability_for_condition(module, 'angina')))
    healthy.transition_set.default_output = healthy

    acute_phase_duration = timedelta(days=28)

    heart_attack.transition_set.add(Transition(mild_heart_failure, _delayed_transition(module, 'mild_heart_failure', 'ihd_event_time', acute_phase_duration)))
    heart_attack.transition_set.add(Transition(moderate_heart_failure, _delayed_transition(module, 'moderate_heart_failure', 'ihd_event_time', acute_phase_duration)))
    heart_attack.transition_set.add(Transition(severe_heart_failure, _delayed_transition(module, 'severe_heart_failure', 'ihd_event_time', acute_phase_duration)))
    heart_attack.transition_set.add(Transition(angina, _delayed_transition(module, 'post_heart_attack_angina', 'ihd_event_time', acute_phase_duration)))

    mild_heart_failure.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'additional_heart_attack')))
    moderate_heart_failure.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'additional_heart_attack')))
    severe_heart_failure.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'additional_heart_attack')))
    angina.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'additional_heart_attack')))

    machine.update([healthy, heart_attack, mild_heart_failure, moderate_heart_failure, severe_heart_failure, angina])

    module.disease_state_machine = machine

    return module

def ihd_factory():
    module = DiseaseModule(
            incidence_rates = {
                'heart_attack': 'ihd_incidence_rates.csv',
                'chronic_ihd': _build_table(1),
            },
            mortality_rates = {
                'healthy': _build_table(0.0),
                'heart_attack': 'mi_acute_excess_mortality.csv',
                'chronic_ihd': 'ihd_mortality_rate.csv',
            }
    )
    machine = Machine('ihd')

    healthy = State('healthy')
    def heart_attack_side_effect(agents, state_column):
        agents[state_column+'_event_time'] = module.simulation.current_time.timestamp()
        agents[state_column+'_event_count'] += 1
        return agents
    heart_attack = State('heart_attack', heart_attack_side_effect)
    chronic_ihd = State('chronic_ihd')

    acute_phase_duration = timedelta(days=28)

    healthy.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'heart_attack')))

    heart_attack.transition_set.add(Transition(chronic_ihd, _delayed_transition(module, 'chronic_ihd', 'ihd_event_time', acute_phase_duration)))

    chronic_ihd.transition_set.add(Transition(heart_attack, _probability_for_condition(module, 'heart_attack')))

    machine.update([healthy, heart_attack, chronic_ihd])
    module.disease_state_machine = machine

    return module

def hemorrhagic_stroke_factory():
    module = DiseaseModule(
            incidence_rates = {
                'hemorrhagic_stroke': 'hem_stroke_incidence_rates.csv',
                'chronic_stroke': _build_table(1),
            },
            mortality_rates = {
                'healthy': _build_table(0.0),
                'hemorrhagic_stroke': 'acute_hem_stroke_excess_mortality.csv',
                'chronic_stroke': 'chronic_hem_stroke_excess_mortality.csv',
            }
    )
    machine = Machine('hemorrhagic_stroke')

    healthy = State('healthy')
    def stroke_side_effect(agents, state_column):
        agents[state_column+'_event_time'] = module.simulation.current_time.timestamp()
        agents[state_column+'_event_count'] += 1
        return agents
    stroke = State('stroke', stroke_side_effect)
    chronic_stroke = State('chronic_stroke')

    acute_phase_duration = timedelta(days=28)

    healthy.transition_set.add(Transition(stroke, _probability_for_condition(module, 'hemorrhagic_stroke')))

    stroke.transition_set.add(Transition(chronic_stroke, _delayed_transition(module, 'chronic_stroke', 'hemorrhagic_stroke_event_time', acute_phase_duration)))

    chronic_stroke.transition_set.add(Transition(stroke, _probability_for_condition(module, 'hemorrhagic_stroke')))

    machine.update([healthy, stroke, chronic_stroke])
    module.disease_state_machine = machine

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

class DiseaseModule(SimulationModule):
    def __init__(self, incidence_rates, mortality_rates):
        super(DiseaseModule, self).__init__()
        self.disease_state_machine = None
        self.incidence_rates_defs = incidence_rates
        self.mortality_rates_defs = mortality_rates

    def module_id(self):
        return (self.__class__, self.disease_state_machine.state_column)

    def setup(self):
        self.register_event_listener(self.transition_handler, 'time_step')
        for label in self.incidence_rates_defs.keys():
            self.register_value_source(partial(self.incidence_rates, label='{0}_incidence'.format(label)), 'incidence_rates', label)

        for label in self.mortality_rates_defs.keys():
            self.register_value_mutator(partial(self.mortality_rates, label=label), 'mortality_rates')

    def transition_handler(self, event):
        affected_population = self.disease_state_machine.transition(event.affected_population)
        self.simulation.population.loc[affected_population.index] = affected_population

    def incidence_rates(self, population, label):
        mediation_factor = self.simulation.incidence_mediation_factor(self.disease_state_machine.state_column)
        return pd.Series(self.lookup_columns(population, [label])[label].values * mediation_factor, index=population.index)

    def mortality_rates(self, population, rates, label):
        column_name = '{0}_mortality'.format(label)
        rates += self.lookup_columns(population, [column_name])[column_name].values * (population[self.disease_state_machine.state_column] == label)
        return rates

    def load_population_columns(self, path_prefix, population_size):
        # TODO: Load real data and integrate with state machine
        self.population_columns = pd.DataFrame(['healthy']*population_size, columns=[self.disease_state_machine.state_column])
        self.population_columns[self.disease_state_machine.state_column + '_event_count'] = 0
        self.population_columns[self.disease_state_machine.state_column + '_event_time'] = np.array([0] * population_size, dtype=np.float)

    def _load_data(self, path_prefix):
        full_lookup_table = pd.DataFrame(columns=['age', 'sex', 'year'])
        for tables, table_type in [(self.incidence_rates_defs, 'incidence'), (self.mortality_rates_defs, 'mortality')]:
            for label, rate_source in tables.items():
                if isinstance(rate_source, str):
                    table_path = os.path.join(path_prefix, rate_source)
                    lookup_table = pd.read_csv(table_path)
                    lookup_table.columns = _rename_rate_column(lookup_table, '{0}_{1}'.format(label, table_type))
                    lookup_table.drop_duplicates(['age', 'year', 'sex'], inplace=True)
                    full_lookup_table = full_lookup_table.merge(lookup_table, on=['age', 'sex', 'year'], how='outer')
                else:
                    rate_source.columns = _rename_rate_column(rate_source, '{0}_{1}'.format(label, table_type))
                    full_lookup_table = full_lookup_table.merge(rate_source, on=['age', 'sex', 'year'], how='outer')
        return full_lookup_table
