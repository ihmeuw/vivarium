import pytest

import pandas as pd

from ceam.framework.population import PopulationManager

def test_column_observers():
    manager = PopulationManager()

    # Setup test population
    manager.growing = True
    population_view = manager.get_view(['A', 'B', 'C'])
    population_view.update(pd.DataFrame({'A': range(10), 'B': range(10,20), 'C': range(30,40)}))
    manager.growing = False

    observed = [0]
    def observe_a():
        observed[0] = manager.population.A.max()

    manager.register_observer('A', observe_a)
    population_view.update(pd.Series(range(40,50), name='A'))
    assert observed[0] == 49


    population_view.update(pd.Series(range(50,60), name='B'))
    assert observed[0] == 49

    population_view.update(pd.DataFrame({'A': range(10), 'C': range(60, 70)}))
    assert observed[0] == 9

    manager.deregister_observer('A', observe_a)
    population_view.update(pd.Series(range(40,50), name='A'))
    assert observed[0] == 9

