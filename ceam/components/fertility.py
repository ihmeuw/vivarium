from datetime import timedelta

import pandas as pd
import numpy as np

from ceam import config

from ceam.framework.util import rate_to_probability, from_yearly

from ceam.components.state_test import Transition, TransitionSet, record_event_time, active_after_delay, new_state_side_effect
from ceam.framework.event import listens_for
from ceam.framework.population import uses_columns, creates_simulants

class Fertility:
    def setup(self, builder):
        time_step = config.getfloat('simulation_parameters', 'time_step')
        time_step = timedelta(days=time_step)

        self.transitions = TransitionSet(builder.randomness('fertility'))
        conception = Transition(self.conception_probability,
                                [record_event_time('last_conception_time', builder.clock()), new_state_side_effect('pregnancy', 'pregnant')]
                                )
        birth      = Transition(1,
                                [record_event_time('last_birth_time', builder.clock()), new_state_side_effect('pregnancy', 'not_pregnant'), self.add_child],
                                [active_after_delay(timedelta(days=9*30.5), 'last_conception_time', builder.clock())]
                                )

        self.transitions.table['not_pregnant'].append(conception)
        self.transitions.table['not_pregnant'].append(Transition.RESIDUAL)
        self.transitions.table['pregnant'].append(birth)
        self.transitions.table['pregnant'].append(Transition.RESIDUAL)

        self.initial_population_view = builder.population_view(['sex'])

        return [self.transitions]

    @listens_for('time_step')
    @uses_columns(['pregnancy'], 'alive == True and sex == "Female"')
    def step(self, event):
        self.transitions.transition(event.population.pregnancy)

    @listens_for('initialize_simulants')
    @uses_columns(['sex', 'fertility'])
    def make_fertility_column(self, event):
        fertility = pd.Series(0.0, name='fertility', index=event.index)
        women = self.initial_population_view.get(event.index).sex == 'Female'
        fertility[women] = np.random.random(size=women.sum())*0.1
        event.population_view.update(fertility)

    @listens_for('initialize_simulants')
    @uses_columns(['pregnancy'])
    def make_pregnancy_column(self, event):
        event.population_view.update(pd.Series('not_pregnant', index=event.index))

    @uses_columns(['fertility'])
    def conception_probability(self, index, population_view):
        population = population_view.get(index)
        time_step = config.getfloat('simulation_parameters', 'time_step')
        time_step = timedelta(days=time_step)
        return rate_to_probability(from_yearly(population.fertility, time_step))

    @creates_simulants
    def add_child(self, index, creator):
        if len(index) > 0:
            creator(len(index), population_configuration={'initial_age': 1})
