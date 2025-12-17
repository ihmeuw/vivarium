# mypy: ignore-errors
import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder
from vivarium.framework.state_machine import Machine, State, Transition
from vivarium.framework.utilities import rate_to_probability
from vivarium.framework.values import list_combiner, union_post_processor
from collections.abc import Iterable

class DiseaseTransition(Transition):
    #####################
    # Lifecycle methods #
    #####################
    def __init__(
        self,
        input_state: "DiseaseState",
        output_state: "DiseaseState",
        cause_key: str,
        measure: str,
        rate_name: str,
        **kwargs,
    ):
        super().__init__(
            input_state, output_state, probability_func=self._probability, **kwargs
        )
        self.cause_key = cause_key
        self.measure = measure
        self.rate_name = rate_name
        self.join_paf_pipeline = f"{self.rate_name}.population_attributable_fraction"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        rate = builder.configuration[self.cause_key][self.measure]

        self.base_rate = lambda index: pd.Series(rate, index=index)
        builder.value.register_attribute_producer(
            self.join_paf_pipeline,
            source=lambda index: [pd.Series(0.0, index=index)],
            component=self,
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )
        builder.value.register_rate_producer(
            self.rate_name,
            source=self._risk_deleted_rate,
            component=self,
            required_resources=[self.join_paf_pipeline],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _risk_deleted_rate(self, index: pd.Index) -> pd.Series:
        joint_paf = self.population_view.get_attributes(index, self.join_paf_pipeline)
        return self.base_rate(index) * (1 - joint_paf)

    ##################
    # Helper methods #
    ##################

    def _probability(self, index: pd.Index) -> pd.Series:
        return pd.Series(rate_to_probability(self.population_view.get_attributes(index, self.rate_name)))


class DiseaseState(State):

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, state_id: str, cause_key: str, with_excess_mortality: bool = False):
        super().__init__(state_id)
        self._cause_key = cause_key
        self._with_excess_mortality = with_excess_mortality
        self.emr_paf_pipeline = f"{self.state_id}.excess_mortality_rate.population_attributable_fraction"
        self.emr_pipeline = f"{self.state_id}.excess_mortality_rate"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        """Performs this component's simulation setup.

        Parameters
        ----------
        builder
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
        builder.value.register_attribute_producer(
            self.emr_paf_pipeline,
            source=lambda index: [pd.Series(0.0, index=index)],
            component=self,
            preferred_combiner=list_combiner,
            preferred_post_processor=union_post_processor,
        )

        builder.value.register_rate_producer(
            self.emr_pipeline,
            source=self.risk_deleted_excess_mortality_rate,
            component=self,
            required_resources=[self.emr_paf_pipeline],
        )

        builder.value.register_attribute_modifier(
            "mortality_rate",
            modifier=self.add_in_excess_mortality,
            component=self,
            required_resources=[self.emr_pipeline]
        )

    ##################
    # Public methods #
    ##################

    def add_disease_transition(
        self, output: "DiseaseState", measure: str, rate_name: str, **kwargs
    ) -> DiseaseTransition:
        t = DiseaseTransition(self, output, self._cause_key, measure, rate_name, **kwargs)
        self.add_transition(t)
        return t

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def risk_deleted_excess_mortality_rate(self, index: pd.Index) -> pd.Series:
        emr_paf = self.population_view.get_attributes(index, self.emr_paf_pipeline)
        return pd.Series(self._excess_mortality_rate, index=index) * (1 - emr_paf)

    def add_in_excess_mortality(
        self, index: pd.Index, mortality_rates: pd.Series
    ) -> pd.Series:
        mortality_rates.loc[index] += self.population_view.get_attributes(
            index, self.emr_pipeline
        )
        return mortality_rates


class DiseaseModel(Machine):

    def __init__(
        self,
        state_column: str,
        states: Iterable[State] = (),
        initial_state: State | None = None,
    ) -> None:
        super().__init__(state_column, states, initial_state)
        self.csmr_pipeline = f"{self.state_column}.cause_specific_mortality_rate"

    #####################
    # Lifecycle methods #
    #####################

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        super().setup(builder)
        config = builder.configuration[self.state_column]
        # Reasonable approximation for short duration diseases.
        case_fatality_rate = config.excess_mortality_rate / (
            config.excess_mortality_rate + config.remission_rate
        )
        cause_specific_mortality_rate = config.incidence_rate * case_fatality_rate

        builder.value.register_rate_producer(
            self.csmr_pipeline,
            source=lambda index: pd.Series(cause_specific_mortality_rate, index=index),
            component=self,
        )
        builder.value.register_attribute_modifier(
            "mortality_rate",
            modifier=self.delete_cause_specific_mortality,
            required_resources=[self.csmr_pipeline],
            component=self,
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def delete_cause_specific_mortality(self, index: pd.Index, rates: pd.Series) -> pd.Series:
        csmr = self.population_view.get_attributes(index, self.csmr_pipeline)
        return rates - csmr


class SISDiseaseModel(Component):
    configuration_defaults = {
        "disease": {
            "incidence_rate": 0.005,
            "remission_rate": 0.05,
            "excess_mortality_rate": 0.01,
        }
    }

    def __init__(self, disease_name: str):
        super().__init__()
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
