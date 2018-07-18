import pandas as pd

from vivarium.framework.state_machine import State, Machine, Transition
from vivarium.framework.util import rate_to_probability
from vivarium.framework.values import list_combiner, joint_value_post_processor


class DiseaseTransition(Transition):
    def __init__(self, name, rate, input_state, output_state, **kwargs):
        super().__init__(input_state, output_state, probability_func=self._probability, **kwargs)
        self.name = name
        self.base_rate = lambda index: pd.Series(rate, index=index)

    def setup(self, builder):
        self.risk_deleted_rate = builder.value.register_rate_producer(f'{self.name}_rate',
                                                                      source=self._risk_deleted_rate)
        self.joint_population_attributable_fraction = builder.value.register_value_producer(
            f'{self.output_state.state_id}.population_attributable_fraction',
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=joint_value_post_processor)

    def _probability(self, index):
        return rate_to_probability(self.risk_deleted_rate(index))

    def _risk_deleted_rate(self, index):
        return self.base_rate(index) * (1 - self.joint_population_attributable_fraction(index))


class DiseaseState(State):

    def __init__(self, state_name, excess_mortality_rate=0, **kwargs):
        super().__init__(state_name, **kwargs)
        self.excess_mortality_rate = excess_mortality_rate

    def setup(self, builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        self.clock = builder.time.clock()

        columns_required = [self._model, 'alive']
        columns_created = [f'{self.state_id}_event_count']

        builder.population.initializes_simulants(self.on_initialize_simulants, creates_columns=columns_created)
        self.population_view = builder.population.get_view(columns_required + columns_created)

        builder.value.register_value_modifier('mortality_rate', self.add_in_excess_mortality)
        builder.value.register_value_modifier('metrics', self.metrics)

    def add_transition(self, transition_name, output, rate=1e6, **kwargs):
        t = DiseaseTransition(transition_name, rate, self, output, **kwargs)
        self.transition_set.append(t)
        return t

    def on_initialize_simulants(self, pop_data):
        self.population_view.update(
            pd.DataFrame({f'{self.state_id}_event_count': pd.Series(0, index=pop_data.index)},
                         index=pop_data.index)
        )

    def add_in_excess_mortality(self, index, mortality_rates):
        pop = self.population_view.get(index)
        affected = pop[self._model] == self.state_id
        mortality_rates[affected] += self.excess_mortality_rate
        return mortality_rates

    def metrics(self, index, metrics):
        """Records data for simulation post-processing.

        Parameters
        ----------
        index : iterable of ints
            An iterable of integer labels for the simulants.
        metrics : `pandas.DataFrame`
            A table for recording simulation events of interest in post-processing.

        Returns
        -------
        `pandas.DataFrame`
            The metrics table updated to reflect new simulation state."""
        population = self.population_view.get(index)
        metrics[f'{self.state_id}_event_count'] = population[f'{self.state_id}_event_count'].sum()

        return metrics


class DiseaseModel(Machine):

    def __init__(self, disease, initial_state, cause_specific_mortality_rate=0., **kwargs):
        super().__init__(disease, **kwargs)
        self.initial_state = initial_state.state_id
        self.cause_specific_mortality_rate = cause_specific_mortality_rate

    def setup(self, builder):
        super().setup(builder)

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
        return rates - self.cause_specific_mortality_rate

    def metrics(self, index, metrics):
        pop = self.population_view.get(index, query="alive == 'alive'")
        metrics[self.state_column + '_prevalent_cases'] = pop[pop[self.state_column] != self.initial_state].sum()
        return metrics


class SIS_DiseaseModel:

    configuration_defaults = {
        'disease': {
            'incidence': 0.005,
            'remission': 0.05,
            'cause_specific_mortality': 0.001,
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
        susceptible_state.add_transition(f'{self.name}.incidence', infected_state, rate=config.incidence)
        infected_state.allow_self_transitions()
        infected_state.add_transition(f'{self.name}.remission', susceptible_state, rate=config.remission)

        model = DiseaseModel(self.name,
                             initial_state=susceptible_state,
                             cause_specific_mortality_rate=config.cause_specific_mortality,
                             states=[susceptible_state, infected_state])
        builder.components.add_components([model])


class SimpleIntervention:

    intervention_group = 'age >= 25 and alive == "alive"'

    def setup(self, builder):
        self.clock = builder.time.clock()
        self.reset()
        builder.value.register_value_modifier('mortality_rate', modifier=self.mortality_rates)
        self.population_view = builder.population.get_view(['age', 'alive'], query=self.intervention_group)
        builder.event.register_listener('time_step', self.track_cost)
        builder.event.register_listener('simulation_end', self.dump_metrics)

    def track_cost(self, event):
        if event.time.year >= 1995:
            time_step = event.step_size
            # FIXME: charge full price once per year?
            self.cumulative_cost += 2.0 * len(event.index) * (time_step / pd.Timedelta(days=365))

    def mortality_rates(self, index, rates):
        if self.clock().year >= 1995:
            pop = self.population_view.get(index)
            rates.loc[pop.index] *= 0.5
        return rates

    def reset(self):
        self.cumulative_cost = 0

    def dump_metrics(self, event):
        print('Cost:', self.cumulative_cost)






