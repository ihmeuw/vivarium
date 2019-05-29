
class Observer:

    configuration_defaults = {
        'mortality': {
            'life_expectancy': 80,
        }
    }

    def __init__(self):
        self.name = 'observer'

    def setup(self, builder):
        self.life_expectancy = builder.configuration.mortality.life_expectancy
        self.population_view = builder.population.get_view(['age', 'alive'])

        builder.value.register_value_modifier('metrics', self.metrics)

    def metrics(self, index, metrics):

        pop = self.population_view.get(index)
        metrics['total_population_alive'] = len(pop[pop.alive == 'alive'])
        metrics['total_population_dead'] = len(pop[pop.alive == 'dead'])

        metrics['years_of_life_lost'] = (self.life_expectancy - pop.age[pop.alive == 'dead']).sum()

        return metrics
