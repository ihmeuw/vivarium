# mypy: ignore-errors
from typing import Any, Dict, List

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder


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
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.risk: self.CONFIGURATION_DEFAULTS["risk"]}

    @property
    def columns_created(self) -> List[str]:
        return [f"{self.risk}_propensity"]

    @property
    def initialization_requirements(self) -> Dict[str, List[str]]:
        return {
            "requires_columns": [],
            "requires_values": [],
            "requires_streams": [self.risk],
        }

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk: str):
        super().__init__()
        self.risk = risk

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        proportion_exposed = builder.configuration[self.risk].proportion_exposed
        self.base_exposure_threshold = builder.value.register_value_producer(
            f"{self.risk}.base_proportion_exposed",
            source=lambda index: pd.Series(proportion_exposed, index=index),
        )
        self.exposure_threshold = builder.value.register_value_producer(
            f"{self.risk}.proportion_exposed", source=self.base_exposure_threshold
        )

        self.exposure = builder.value.register_value_producer(
            f"{self.risk}.exposure", source=self._exposure
        )
        self.randomness = builder.randomness.get_stream(self.risk)

    ########################
    # Event-driven methods #
    ########################

    def on_initialize_simulants(self, pop_data):
        draw = self.randomness.get_draw(pop_data.index)
        self.population_view.update(pd.Series(draw, name=f"{self.risk}_propensity"))

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def _exposure(self, index):
        propensity = self.population_view.get(index)[f"{self.risk}_propensity"]
        return self.exposure_threshold(index) > propensity


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
    def configuration_defaults(self) -> Dict[str, Any]:
        return {self.risk: self.CONFIGURATION_DEFAULTS["risk_effect"]}

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, risk_name: str, disease_rate: str):
        super().__init__()
        self.risk_name = risk_name
        self.disease_rate = disease_rate
        self.risk = f"effect_of_{risk_name}_on_{disease_rate}"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        relative_risk = builder.configuration[self.risk].relative_risk
        self.relative_risk = builder.value.register_value_producer(
            f"{self.risk}.relative_risk",
            source=lambda index: pd.Series(relative_risk, index=index),
        )

        builder.value.register_value_modifier(
            f"{self.disease_rate}.population_attributable_fraction",
            self.population_attributable_fraction,
        )
        builder.value.register_value_modifier(f"{self.disease_rate}", self.rate_adjustment)
        self.base_risk_exposure = builder.value.get_value(
            f"{self.risk_name}.base_proportion_exposed"
        )
        self.actual_risk_exposure = builder.value.get_value(f"{self.risk_name}.exposure")

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def population_attributable_fraction(self, index):
        exposure = self.base_risk_exposure(index)
        relative_risk = self.relative_risk(index)
        return exposure * (relative_risk - 1) / (exposure * (relative_risk - 1) + 1)

    def rate_adjustment(self, index, rates):
        exposed = self.actual_risk_exposure(index)
        rr = self.relative_risk(index)
        rates[exposed] *= rr[exposed]
        return rates
