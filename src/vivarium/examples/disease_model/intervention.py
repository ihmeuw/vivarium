import pandas as pd

from vivarium.framework.engine import Builder


class TreatmentIntervention:

    configuration_defaults = {
        "intervention": {
            "effect_size": 0.5,
        }
    }

    def __init__(self, name: str, affected_value: str):
        self.name = name
        self.affected_value = affected_value
        self.configuration_defaults = {
            name: TreatmentIntervention.configuration_defaults["intervention"]
        }

    # noinspection PyAttributeOutsideInit
    def setup(self, builder: Builder):
        effect_size = builder.configuration[self.name].effect_size
        builder.value.register_value_modifier(
            self.affected_value, modifier=self.intervention_effect
        )
        self.effect_size = builder.value.register_value_producer(
            f"{self.name}.effect_size",
            source=lambda index: pd.Series(effect_size, index=index),
        )

    def intervention_effect(self, index, value):
        return value * (1 - self.effect_size(index))
