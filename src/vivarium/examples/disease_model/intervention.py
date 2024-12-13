# mypy: ignore-errors
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

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder) -> None:
        effect_size = builder.configuration[self.intervention].effect_size
        self.effect_size = builder.value.register_value_producer(
            f"{self.intervention}.effect_size",
            source=lambda index: pd.Series(effect_size, index=index),
        )
        builder.value.register_value_modifier(
            self.affected_value,
            modifier=self.intervention_effect,
            required_resources=[self.effect_size],
        )

    ##################################
    # Pipeline sources and modifiers #
    ##################################

    def intervention_effect(self, index: pd.Index, value: pd.Series) -> pd.Series:
        return value * (1 - self.effect_size(index))
