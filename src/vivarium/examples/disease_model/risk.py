import pandas as pd


class Risk:

    configuration_defaults = {
        'risk': {
            'proportion_exposed': 0.3,
        },
    }

    def __init__(self, name):
        self.name = name
        self.configuration_defaults = {name: Risk.configuration_defaults['risk']}

    def setup(self, builder):
        proportion_exposed = builder.configuration[self.name].proportion_exposed
        self.base_exposure_threshold = builder.value.register_value_producer(
            f'{self.name}.base_proportion_exposed', source=lambda index: pd.Series(proportion_exposed, index=index)
        )
        self.exposure_threshold = builder.value.register_value_producer(
            f'{self.name}.proportion_exposed', source=self.base_exposure_threshold
        )

        self.exposure = builder.value.register_value_producer(f'{self.name}.exposure', source=self._exposure)
        self.randomness = builder.randomness.get_stream(self.name)

        columns_created = [f'{self.name}_propensity']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=columns_created,
                                                 requires_streams=[self.name])
        self.population_view = builder.population.get_view(columns_created)

    def on_initialize_simulants(self, pop_data):
        draw = self.randomness.get_draw(pop_data.index)
        self.population_view.update(pd.Series(draw, name=f'{self.name}_propensity'))

    def _exposure(self, index):
        propensity = self.population_view.get(index)[f'{self.name}_propensity']
        return self.exposure_threshold(index) > propensity

    def __repr__(self):
        return f"Risk(name={self.name})"


class DirectEffect:

    configuration_defaults = {
        'direct_effect': {
            'relative_risk': 2,
        },
    }

    def __init__(self, risk, disease_rate):
        self.risk = risk
        self.disease_rate = disease_rate
        self.name = f'effect_of_{risk}_on_{disease_rate}'
        self.configuration_defaults = {self.name: DirectEffect.configuration_defaults['direct_effect']}

    def setup(self, builder):
        relative_risk = builder.configuration[self.name].relative_risk
        self.relative_risk = builder.value.register_value_producer(
            f'{self.name}.relative_risk', source=lambda index: pd.Series(relative_risk, index=index)
        )

        builder.value.register_value_modifier(f'{self.disease_rate}.population_attributable_fraction',
                                              self.population_attributable_fraction)
        builder.value.register_value_modifier(f'{self.disease_rate}',
                                              self.rate_adjustment)
        self.base_risk_exposure = builder.value.get_value(f'{self.risk}.base_proportion_exposed')
        self.actual_risk_exposure = builder.value.get_value(f'{self.risk}.exposure')

    def population_attributable_fraction(self, index):
        exposure = self.base_risk_exposure(index)
        relative_risk = self.relative_risk(index)
        return exposure * (relative_risk - 1) / (exposure * (relative_risk - 1) + 1)

    def rate_adjustment(self, index, rates):
        exposed = self.actual_risk_exposure(index)
        rr = self.relative_risk(index)
        rates[exposed] *= rr[exposed]
        return rates
