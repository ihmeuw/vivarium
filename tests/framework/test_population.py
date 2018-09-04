import pandas as pd
import pytest

from vivarium.framework.population import PopulationView, PopulationManager, PopulationError


class DummyPopulationManager:
    def __init__(self):
        self.population = pd.DataFrame({'age': [0, 10, 20, 30, 40, 50, 60, 70], 'sex': ['Male', 'Female']*4})


def test_create_PopulationView_with_all_columns():
    manager = DummyPopulationManager()
    view = PopulationView(manager)
    assert set(view.columns) == {'age', 'sex'}


def test_circular_initializers():
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer 1",
                                          ['init_1_column'], ['init_2_column'])
    manager.register_simulant_initializer(lambda: "initializer 2",
                                          ['init_2_column'], ['init_1_column'])
    with pytest.raises(PopulationError, match="Check for cyclic dependencies"):
        manager._order_initializers()


def test_missing_initializer():
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer 1",
                                          ["result_column"], ["input_column"])
    with pytest.raises(PopulationError, match="Check for missing dependencies"):
        manager._order_initializers()


def test_conflicting_initializers():
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer 1",
                                          ['result_column'], [])
    manager.register_simulant_initializer(lambda: "initializer 2",
                                          ['result_column'], [])
    with pytest.raises(PopulationError, match="Multiple components are attempting "
                                              "to initialize the same columns"):
        manager._order_initializers()