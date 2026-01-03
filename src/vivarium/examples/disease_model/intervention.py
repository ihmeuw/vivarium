from __future__ import annotations

from typing import Any

import pandas as pd

from vivarium import Component
from vivarium.framework.engine import Builder


class TreatmentIntervention(Component):
    CONFIGURATION_DEFAULTS: dict[str, Any] = {
        "intervention": {
            "effect_size": 0.5,
        }
    }

    ##############
    # Properties #
    ##############

    @property
    def configuration_defaults(self) -> dict[str, Any]:
        return {self.intervention: self.CONFIGURATION_DEFAULTS["intervention"]}

    #####################
    # Lifecycle methods #
    #####################

    def __init__(self, intervention: str, affected_value: str):
        super().__init__()
        self.intervention = intervention
        self.affected_value = affected_value
        self.effect_size_pipeline = f"{self.intervention}.effect_size"

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        effect_size = builder.configuration[self.intervention].effect_size
        builder.value.register_attribute_producer(
            self.effect_size_pipeline,
            source=lambda index: pd.Series(effect_size, index=index),
        )
        builder.value.register_attribute_modifier(
            self.affected_value,
            modifier=self.intervention_effect,
            required_resources=[self.effect_size_pipeline],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def intervention_effect(self, index: pd.Index[int], value: pd.Series[float]) -> pd.Series[float]:
        effect_size = self.population_view.get_attributes(index, self.effect_size_pipeline)
        return value * (1 - effect_size)
