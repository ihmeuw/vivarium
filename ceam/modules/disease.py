import pandas as pd
import numpy as np

from ceam.tree import Node
from ceam.util import filter_for_rate
from ceam.state_machine import Machine, State, GatewayState
from ceam.engine import SimulationModule, DataLoaderMixin

class D
class HealthyState(State, DataLoaderMixin, ValueMutationNode):
    state_id = 'healthy'
    def __init__(self, transition, incidence_table_name):
        State.[transition])
        self.incidence_table_name = incidence_tabel_name

    def transition(self, agents, state_column):
        incidence_rates = self.root.incidence_rates(affected_population, self.condition)
        affected_population = filter_for_rate(agents, incidence_rates)
        affected_population[state_colum] = self.transitions[0]
        affected_population[self.parent.condition + '_event_time'] = self.root.current_time.timestamp()
        affected_population[self.parent.condition + '_event_count'] = 1
        return affected_population

    def _load_data(self, path_prefix):
        return pd.read_csv(os.path.join(path_prefix, self.incidence_table_name)).rename(columns=lambda col: col.lower())

class AcutePhaseState(GatewayState, ValueMutationNode):
    state_id = 'acute'
    def __init__(self, transition, mortality_table_name, phase_duration=28):
        super(AcutePhaseState, self).__init__(transition)
        self.phase_duration = phase_duration
        self.mortality_table_name = mortality_table_name

    def _load_data(self, path_prefix):
        return pd.read_csv(os.path.join(path_prefix, self.mortality_table_name)).rename(columns=lambda col: col.lower())

    def gateway(agents):
        return agents.loc[agents[self.parent.condition+'_event_time'] + self.phase_duration <= self.root.current_time]

class DiseaseState(ChoiceState):
    def __init__(self, mortality_table_name, disability_weight, transitions, dwell_time):
        super(DiseaseState, self).__init__(transitions, [lambda agents: self.transition_condition(agents, table) for table in transition_tables])
        self.mortality_table_name = mortality_table_name
        self.disability_weight = disability_weight
        self.transition_tables = transition_tables

    def _load_data(self, path_prefix):
        lookup_table = pd.DataFrame()
        for table in transition_tables:
            table_path = os.path.join(path_prefix, 
        return lookup_table

m = DiseaseModule('ihd')
m.add_state()
def DiseaseModule(SimulationModule):
    def __init__(self, condition):
        super(DiseaseModule, self).__init__()
        self.condition = condition
        self.machine = Machine(condition, [])

    def _load_data(self, path_prefix):
        lookup_table = pd.DataFrame()
        for state in self.machine.states:
            lookup_table = lookup_table.merge(state.load_data(path_prefix))
