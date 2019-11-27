import pandas as pd
import pytest

from vivarium.framework.population import PopulationView, InitializerComponentSet, PopulationError, PopulationManager


class DummyPopulationManager:
    def __init__(self):
        self.get_population = lambda _ : pd.DataFrame({'age': [0, 10, 20, 30, 40, 50, 60, 70], 'sex': ['Male', 'Female']*4})


def test_create_PopulationView_with_all_columns():
    manager = DummyPopulationManager()
    view = PopulationView(manager, 0)
    assert set(view.columns) == {'age', 'sex'}


def test_initializer_set_fail_type():
    component_set = InitializerComponentSet()

    with pytest.raises(TypeError):
        component_set.add(lambda: 'test', ['test_column'])

    def initializer():
        return 'test'

    with pytest.raises(TypeError):
        component_set.add(initializer, ['test_column'])


class UnnamedComponent:

    def initializer(self):
        return 'test'


class Component:

    def __init__(self, name):
        self.name = name

    def initializer(self):
        return 'test'

    def other_initializer(self):
        return 'whoops'


def test_initializer_set_fail_attr():
    component_set = InitializerComponentSet()

    with pytest.raises(AttributeError):
        component_set.add(UnnamedComponent().initializer, ['test_column'])


def test_initializer_set_duplicate_component():
    component_set = InitializerComponentSet()
    component = Component('test')

    component_set.add(component.initializer, ['test_column1'])
    with pytest.raises(PopulationError, match='multiple population initializers'):
        component_set.add(component.other_initializer, ['test_column2'])


def test_initializer_set_duplicate_columns():
    component_set = InitializerComponentSet()
    component1 = Component('test1')
    component2 = Component('test2')
    columns = ['test_column']

    component_set.add(component1.initializer, columns)
    with pytest.raises(PopulationError, match='both registered initializers'):
        component_set.add(component2.initializer, columns)

    with pytest.raises(PopulationError, match='both registered initializers'):
        component_set.add(component2.initializer, ['sneaky_column'] + columns)


def test_initializer_set():
    component_set = InitializerComponentSet()
    for i in range(10):
        component = Component(i)
        columns = [f'test_column_{i}_{j}' for j in range(5)]
        component_set.add(component.initializer, columns)


def test_get_view_with_no_query():
    manager = PopulationManager()
    view = manager._get_view(columns=['age','sex'])
    assert view.query == 'tracked == True'
