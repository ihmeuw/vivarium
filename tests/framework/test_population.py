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


def make_initializer_dag(num_nodes, edges):
    """Helper to create a dag organized as a list of dictionaries from
    a node count and edges defined as tuples of (node_index, column_name)

    Always assigns created column as "Column{index}"
    """
    dag = [{'name': f"initializer{i}",
            'creates': [f"Column{i}"],
            'requires': []} for i in range(num_nodes)]
    for edge in edges:
        node_index, column_name = edge
        dag[node_index]['requires'].append(column_name)

    return dag


@pytest.mark.parametrize("num_nodes,edges", [
    (5, [(0, "Column1"), (1, "Column2"), (2, "Column3"), (2, "Column4"), (4, "Column1")])
    ])
def test_cyclic_dependencies(num_nodes, edges):
    manager = PopulationManager()

    dag = make_initializer_dag(num_nodes, edges)
    for node in dag:
        manager.register_simulant_initializer(lambda: node['name'], 
                                                      node['creates'],
                                                      node['requires'])

    with pytest.raises(PopulationError, match="Check for cyclic dependencies"):
        manager._order_initializers()


@pytest.mark.parametrize("num_nodes,edges", [
    (5, [(0, "Column1"), (0, "Column2"), (2, "Column3"), (2, "Column4"), (0, "NaNColumn")]),
    (5, [(0, "Column2"), (1, "Column2"), (2, "NaNColumn"), (1, "Column3"), (3, "Column4"), (3, "Column1")])
    ])
def test_missing_dependencies(num_nodes, edges):
    manager = PopulationManager()

    dag = make_initializer_dag(num_nodes, edges)
    for node in dag:
        manager.register_simulant_initializer(lambda: node['name'], 
                                                      node['creates'],
                                                      node['requires'])

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
