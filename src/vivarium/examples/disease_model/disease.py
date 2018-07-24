import pandas as pd

from vivarium.framework.state_machine import State, Machine, Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor


class DiseaseTransition(Transition):

    def __init__(self, name, rate, input_state, output_state, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self.name = name
        self.base_rate = lambda index: pd.Series(rate, index=index)

    def setup(self, builder):
        self.transition_rate = builder.value.register_rate_producer(f'{self.name}_rate',
                                                                      source=self._risk_deleted_rate)
        self.joint_population_attributable_fraction = builder.value.register_value_producer(
            f'{self.name}_rate.population_attributable_fraction',
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=joint_value_post_processor)

    def _probability(self, index):
        effective_rate = self.transition_rate(index)
        return rate_to_probability(effective_rate)

    def _risk_deleted_rate(self, index):
        return self.base_rate(index) * (1 - self.joint_population_attributable_fraction(index))


class DiseaseState(State):

    def __init__(self, state_name, excess_mortality_rate=0, **kwargs):
        super().__init__(state_name, **kwargs)
        self._excess_mortality_rate = excess_mortality_rate

    def setup(self, builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.clock = builder.time.clock()

        self.excess_mortality_rate = builder.value.register_rate_producer(
            f'{self.state_id}.excess_mortality_rate',
            source=self.risk_deleted_excess_mortality_rate
        )
        self.excess_mortality_rate_paf = builder.value.register_value_producer(
            f'{self.state_id}.excess_mortality_rate.population_attributable_fraction',
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=joint_value_post_processor
        )

        builder.value.register_value_modifier('mortality_rate', self.add_in_excess_mortality)
        self.population_view = builder.population.get_view(
            [self._model], query=f"alive == 'alive' and {self._model} == '{self.state_id}'")

    def add_transition(self, transition_name, output, rate=1e6, **kwargs):
        t = DiseaseTransition(transition_name, rate, self, output, **kwargs)
        self.transition_set.append(t)
        return t

    def risk_deleted_excess_mortality_rate(self, index):
        return pd.Series(self._excess_mortality_rate, index=index) * (1 - self.excess_mortality_rate_paf(index))

    def add_in_excess_mortality(self, index, mortality_rates):
        affected = self.population_view.get(index)
        mortality_rates.loc[affected.index] += self.excess_mortality_rate(affected.index)

        return mortality_rates


class DiseaseModel(Machine):

    def __init__(self, disease, initial_state, cause_specific_mortality_rate=0., **kwargs):
        super().__init__(disease, **kwargs)
        self.initial_state = initial_state.state_id
        self._cause_specific_mortality_rate = cause_specific_mortality_rate

    def setup(self, builder):
        super().setup(builder)
        self.cause_specific_mortality_rate = builder.value.register_rate_producer(
            f'{self.state_column}.cause_specific_mortality_rate',
            source=lambda index: pd.Series(self._cause_specific_mortality_rate, index=index)
        )
        builder.value.register_value_modifier('mortality_rate', modifier=self.delete_cause_specific_mortality)
        builder.value.register_value_modifier('metrics', modifier=self.metrics)

        creates_columns = [self.state_column]
        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=creates_columns)
        self.population_view = builder.population.get_view(['age', 'sex', self.state_column])

        builder.event.register_listener('time_step', self.on_time_step)

    def on_initialize_simulants(self, pop_data):
        condition_column = pd.Series(self.initial_state, index=pop_data.index, name=self.state_column)
        self.population_view.update(condition_column)

    def on_time_step(self, event):
        self.transition(event.index, event.time)

    def delete_cause_specific_mortality(self, index, rates):
        return rates - self.cause_specific_mortality_rate(index)

    def metrics(self, index, metrics):
        pop = self.population_view.get(index, query="alive == 'alive'")
        metrics[self.state_column + '_prevalent_cases'] = len(pop[pop[self.state_column] != self.initial_state])
        return metrics


class SIS_DiseaseModel:

    configuration_defaults = {
        'disease': {
            'incidence': 0.005,
            'remission': 0.05,
            'excess_mortality': 0.01,
        }
    }

    def __init__(self, disease_name):
        self.name = disease_name
        self.configuration_defaults = {disease_name: SIS_DiseaseModel.configuration_defaults['disease']}

    def setup(self, builder):
        config = builder.configuration[self.name]

        susceptible_state = DiseaseState(f'susceptible_to_{self.name}')
        infected_state = DiseaseState(f'infected_with_{self.name}',
                                      excess_mortality_rate=config.excess_mortality)

        susceptible_state.allow_self_transitions()
        susceptible_state.add_transition(f'{infected_state.state_id}.incidence', infected_state, rate=config.incidence)
        infected_state.allow_self_transitions()
        infected_state.add_transition(f'{infected_state.state_id}.remission', susceptible_state, rate=config.remission)

        # Reasonable approximation for short duration diseases.
        case_fatality_rate = config.excess_mortality / (config.excess_mortality + config.remission)
        cause_specific_mortality = config.incidence * case_fatality_rate

        model = DiseaseModel(self.name,
                             initial_state=susceptible_state,
                             cause_specific_mortality_rate=cause_specific_mortality,
                             states=[susceptible_state, infected_state])
        builder.components.add_components([model])
