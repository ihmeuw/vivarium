from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd

from vivarium import Component

if TYPE_CHECKING:
    from vivarium.framework.engine import Builder
    from vivarium.framework.population import SimulantData


class Risk(Component):
    CONFIGURATION_DEFAULTS = {
        "risk": {
            "proportion_exposed": 0.3,
        },
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {self.risk: self.CONFIGURATION_DEFAULTS["risk"]}

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        super().__init__()
        self.risk = risk
        self.propensity_column = f"{risk}_propensity"
        self.base_proportion_exposed_pipeline = f"{risk}.base_proportion_exposed"
        self.exposure_threshold_pipeline = f"{self.risk}.proportion_exposed"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        proportion_exposed = builder.configuration[self.risk].proportion_exposed
        builder.value.register_attribute_producer(
            self.base_proportion_exposed_pipeline,
            source=lambda index: pd.Series(proportion_exposed, index=index),
        )
        builder.value.register_attribute_producer(
            self.exposure_threshold_pipeline,
            source=[self.base_proportion_exposed_pipeline],
        )

        builder.value.register_attribute_producer(
            f"{self.risk}.exposure",
            source=self._exposure,
            required_resources=[self.propensity_column, self.exposure_threshold_pipeline],
        )
        self.randomness = builder.randomness.get_stream(self.risk)
        builder.population.register_initializer(self.propensity_column, self.on_initialize_simulants, [self.randomness])

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data: SimulantData) -> None:
        draw = self.randomness.get_draw(pop_data.index)
        self.population_view.update(pd.Series(draw, name=self.propensity_column))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _exposure(self, index: pd.Index[int]) -> pd.Series[bool]:
        propensity = self.population_view.get_attributes(index, self.propensity_column)
        exposure_threshold = self.population_view.get_attributes(index, self.exposure_threshold_pipeline)
        return exposure_threshold > propensity


class RiskEffect(Component):
    CONFIGURATION_DEFAULTS = {
        "risk_effect": {
            "relative_risk": 2,
        },
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {self.risk: self.CONFIGURATION_DEFAULTS["risk_effect"]}

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk_name: str, disease_rate: str):
        super().__init__()
        self.risk_name = risk_name
        self.disease_rate = disease_rate
        self.risk = f"effect_of_{risk_name}_on_{disease_rate}"
        self.base_exposure_pipeline = f"{self.risk_name}.base_proportion_exposed"
        self.exposure_pipeline = f"{self.risk_name}.exposure"
        self.relative_risk_pipeline = f"{self.risk}.relative_risk"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        relative_risk = builder.configuration[self.risk].relative_risk
        builder.value.register_attribute_producer(
            self.relative_risk_pipeline,
            source=lambda index: pd.Series(relative_risk, index=index),
        )

        builder.value.register_attribute_modifier(
            f"{self.disease_rate}.population_attributable_fraction",
            modifier=self.population_attributable_fraction,
            required_resources=[self.base_exposure_pipeline, self.relative_risk_pipeline],
        )
        builder.value.register_attribute_modifier(
            f"{self.disease_rate}",
            modifier=self.rate_adjustment,
            required_resources=[self.exposure_pipeline, self.relative_risk_pipeline],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def population_attributable_fraction(self, index: pd.Index[int]) -> pd.Series[float]:
        pop = self.population_view.get_attributes(
            index, [self.base_exposure_pipeline, self.relative_risk_pipeline]
        )
        exposure = pop[self.base_exposure_pipeline]
        relative_risk = pop[self.relative_risk_pipeline]
        return exposure * (relative_risk - 1) / (exposure * (relative_risk - 1) + 1)

    def rate_adjustment(self, index: pd.Index[int], rates: pd.Series[float]) -> pd.Series[float]:
        exposed = self.population_view.get_attributes(index, self.exposure_pipeline)
        rr = self.population_view.get_attributes(index, self.relative_risk_pipeline)
        rates[exposed] *= rr[exposed]
        return rates
