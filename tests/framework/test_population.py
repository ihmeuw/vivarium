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
    """   [4]
          /
        [3]
        / \
      [2]-[5]
      /
    [1]
    """
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer1",
                                          ['Column1'], ['Column2'])
    manager.register_simulant_initializer(lambda: "initializer2",
                                          ['Column2'], ['Column3'])
    manager.register_simulant_initializer(lambda: "initializer 3",
                                          ['Column3'], ['Column4', 'Column5'])
    manager.register_simulant_initializer(lambda: "initializer 4",
                                          ['Column4'], [])
    manager.register_simulant_initializer(lambda: "initializer 5",
                                          ['Column5'], ['Column2'])
    with pytest.raises(PopulationError, match="Check for cyclic dependencies"):
        manager._order_initializers()


def test_missing_initializer():
    """        /
         [4] [5]
          \  /   
      [2] [3]
       \  /
       [1]
    """
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer1",
                                          ['Column1'], ['Column2', 'Column3'])
    manager.register_simulant_initializer(lambda: "initializer2",
                                          ['Column2'], [])
    manager.register_simulant_initializer(lambda: "initializer3",
                                          ['Column3'], ['Column4', 'Column5'])
    manager.register_simulant_initializer(lambda: "initializer4",
                                          ['Column4'], [])
    manager.register_simulant_initializer(lambda: "initializer5",
                                          ['Column5'], ['NonExistantColumn'])
    with pytest.raises(PopulationError, match="Check for missing dependencies"):
        manager._order_initializers()


def test_circular_and_missing_initializer():
    """ [3]--
         |
        [2]
        / \
      [1]-[4]
           |
          [5]
    """
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer1",
                                          ['Column1'], ['Column2'])
    manager.register_simulant_initializer(lambda: "initializer2",
                                          ['Column2'], ['Column3', 'Column4'])
    manager.register_simulant_initializer(lambda: "initializer3",
                                          ['Column3'], ['NonExistantColumn'])
    manager.register_simulant_initializer(lambda: "initializer4",
                                          ['Column4'], ['Column1', 'Column5'])
    manager.register_simulant_initializer(lambda: "initializer5",
                                          ['Column5'], [])
    with pytest.raises(PopulationError, match="Check for missing dependencies"):
        manager._order_initializers()


def test_simple_conflicting_initializers():
    manager = PopulationManager()
    manager.register_simulant_initializer(lambda: "initializer 1",
                                          ['Column1'], [])
    manager.register_simulant_initializer(lambda: "initializer 1",
                                          ['Column1'], [])
    with pytest.raises(PopulationError, match="Multiple components are attempting "
                                              "to initialize the same columns"):
        manager._order_initializers()