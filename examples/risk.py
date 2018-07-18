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
        self._proportion_exposed = builder.configuration[self.name].proportion_exposed

        columns_created = [f'{self.name}_exposure', f'{self.name}_propensity']
        builder.population.initializes_simulants(self.on_initialize_simulants,
                                                 creates_columns=[f'{self.name}_exposure', f'{self.name}_propensity'])

        self.exposure = builder.value.register_value_producer(f'{self.name}.exposure',
                                                              source=self._proportion_exposed)
        self.randomness = builder.randomness.get_stream(self.name)

    def on_initialize_simulants(self, pop_data):
        propensity = self.randomness.get_draw(pop_data.index)
        exposure = self._get_current_exposure(propensity)
        self.population_view.update(pd.DataFrame({
            self._risk+'_propensity': propensity,
            self._risk+'_exposure': exposure,
        }))

    def _get_current_exposure(self, propensity):
        exposure = self.exposure(propensity.index)

        # Get a list of sorted category names (e.g. ['cat1', 'cat2', ..., 'cat9', 'cat10', ...])
        categories = sorted([column for column in exposure if 'cat' in column])
        sorted_exposures = exposure[categories]
        exposure_sum = sorted_exposures.cumsum(axis='columns')
        # Sometimes all data is 0 for the category exposures.  Set the "no exposure" category to catch this case.
        exposure_sum[categories[-1]] = 1  # TODO: Something better than this.

        category_index = (exposure_sum.T < propensity).T.sum('columns')

        return pd.Series(np.array(categories)[category_index], name=self._risk+'_exposure', index=propensity.index)

    def update_exposure(self, event):
        pop = self.population_view.get(event.index)

        propensity = pop[self._risk+'_propensity']
        categories = self._get_current_exposure(propensity)
        self.population_view.update(categories)

    def __repr__(self):
        return f"CategoricalRiskComponent(_risk_type= {self._risk_type}, _risk= {self._risk})"
