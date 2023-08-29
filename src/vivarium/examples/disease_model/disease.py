from typing import Any, Dict, List, Optional

import pandas as pd

from vivarium.framework.engine import Builder
from vivarium.framework.event import Event
from vivarium.framework.population import SimulantData
from vivarium.framework.state_machine import Machine, State, Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, union_post_processor


class DiseaseTransition(Transition):
    def __init__(self, input_state, output_state, cause_key, measure, rate_name, **kwargs):
        super().__init__(
            input_state, output_state, probability_func=self._probability, **kwargs
        )
        self.cause_key = cause_key
        self.measure = measure
        self.rate_name = rate_name

    # noinspection PyAttributeOutsideInit
    def setup(self, builder):
        rate = builder.configuration[self.cause_key][self.measure]

        self.base_rate = lambda index: pd.Series(rate, index=index)
        self.transition_rate = builder.value.register_rate_producer(
            self.rate_name, source=self._risk_deleted_rate
        )
        self.joint_population_attributable_fraction = builder.value.register_value_producer(
            f"{self.rate_name}.population_attributable_fraction",
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

    def _probability(self, index):
        effective_rate = self.transition_rate(index)
        return rate_to_probability(effective_rate)

    def _risk_deleted_rate(self, index):
        return self.base_rate(index) * (
            1 - self.joint_population_attributable_fraction(index)
        )


class DiseaseState(State):

    ##############
    # Properties #
    ##############

    @property
    def population_view_query(self) -> Optional[str]:
        return f"alive == 'alive' and {self.model} == '{self.state_id}'"

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, state_id: str, cause_key: str, with_excess_mortality: bool = False):
        super().__init__(state_id)
        self._cause_key = cause_key
        self._with_excess_mortality = with_excess_mortality

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder : `engine.Builder`
            Interface to several simulation tools.
        """
        super().setup(builder)
        if self._with_excess_mortality:
            self._excess_mortality_rate = builder.configuration[
                self._cause_key
            ].excess_mortality_rate
        else:
            self._excess_mortality_rate = 0

        self.clock = builder.time.clock()

        self.excess_mortality_rate = builder.value.register_rate_producer(
            f"{self.state_id}.excess_mortality_rate",
            source=self.risk_deleted_excess_mortality_rate,
        )
        self.excess_mortality_rate_paf = builder.value.register_value_producer(
            f"{self.state_id}.excess_mortality_rate.population_attributable_fraction",
            source=lambda index: [pd.Series(0, index=index)],
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

        builder.value.register_value_modifier("mortality_rate", self.add_in_excess_mortality)

    ##################
    # Public methods #
    ##################

    def add_disease_transition(
        self, output: State, measure: str, rate_name: str, **kwargs
    ) -> DiseaseTransition:
        t = DiseaseTransition(self, output, self._cause_key, measure, rate_name, **kwargs)
        self.add_transition(t)
        return t

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def risk_deleted_excess_mortality_rate(self, index: pd.Index) -> pd.Series:
        return pd.Series(self._excess_mortality_rate, index=index) * (
            1 - self.excess_mortality_rate_paf(index)
        )

    def add_in_excess_mortality(
        self, index: pd.Index, mortality_rates: pd.Series
    ) -> pd.Series:
        affected = self.population_view.get(index)
        mortality_rates.loc[affected.index] += self.excess_mortality_rate(affected.index)

        return mortality_rates


class DiseaseModel(Machine):

    ##############
    # Properties #
    ##############

    @property
    def columns_created(self) -> List[str]:
        return [self.state_column]

    @property
    def columns_required(self) -> Optional[List[str]]:
        return ["age", "sex"]

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, disease: str, initial_state: DiseaseState, **kwargs):
        super().__init__(disease, **kwargs)
        self.initial_state = initial_state.state_id

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        config = builder.configuration[self.state_column]
        # Reasonable approximation for short duration diseases.
        case_fatality_rate = config.excess_mortality_rate / (
            config.excess_mortality_rate + config.remission_rate
        )
        cause_specific_mortality_rate = config.incidence_rate * case_fatality_rate

        self.cause_specific_mortality_rate = builder.value.register_rate_producer(
            f"{self.state_column}.cause_specific_mortality_rate",
            source=lambda index: pd.Series(cause_specific_mortality_rate, index=index),
        )
        builder.value.register_value_modifier(
            "mortality_rate", modifier=self.delete_cause_specific_mortality
        )
        builder.value.register_value_modifier("metrics", modifier=self.metrics)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData):
        condition_column = pd.Series(
            self.initial_state, index=pop_data.index, name=self.state_column
        )
        self.population_view.update(condition_column)

    def on_time_step(self, event: Event):
        self.transition(event.index, event.time)

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def delete_cause_specific_mortality(self, index: pd.Index, rates: pd.Series):
        return rates - self.cause_specific_mortality_rate(index)

    def metrics(self, index: pd.Index, metrics: Dict[str, Any]):
        pop = self.population_view.get(index, query="alive == 'alive'")
        metrics[self.state_column + "_prevalent_cases"] = len(
            pop[pop[self.state_column] != self.initial_state]
        )
        return metrics


class SISDiseaseModel:
    configuration_defaults = {
        "disease": {
            "incidence_rate": 0.005,
            "remission_rate": 0.05,
            "excess_mortality_rate": 0.01,
        }
    }

    def __init__(self, disease_name):
        self._name = disease_name
        self.configuration_defaults = {
            disease_name: SISDiseaseModel.configuration_defaults["disease"]
        }

        susceptible_state = DiseaseState(f"susceptible_to_{self._name}", self._name)
        infected_state = DiseaseState(
            f"infected_with_{self._name}", self._name, with_excess_mortality=True
        )

        susceptible_state.allow_self_transitions()
        susceptible_state.add_disease_transition(
            infected_state,
            measure="incidence_rate",
            rate_name=f"{infected_state.state_id}.incidence_rate",
        )
        infected_state.allow_self_transitions()
        infected_state.add_disease_transition(
            susceptible_state,
            measure="remission_rate",
            rate_name=f"{infected_state.state_id}.remission_rate",
        )

        model = DiseaseModel(
            self._name,
            initial_state=susceptible_state,
            states=[susceptible_state, infected_state],
        )
        self._sub_components = [model]

    @property
    def name(self):
        return self._name

    @property
    def sub_components(self):
        return self._sub_components
