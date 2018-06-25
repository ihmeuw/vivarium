import pandas as pd

from vivarium.framework.population import PopulationView


class DummyPopulationManager:
    def __init__(self):
        self.population = pd.DataFrame({'age': [0, 10, 20, 30, 40, 50, 60, 70], 'sex': ['Male', 'Female']*4})


def test_create_PopulationView_with_all_columns():
    manager = DummyPopulationManager()
    view = PopulationView(manager)
    assert set(view.columns) == {'age', 'sex'}
