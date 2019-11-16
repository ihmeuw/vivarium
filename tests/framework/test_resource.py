import networkx as nx
import pytest

from vivarium.framework.resource import (ResourceGroup, ResourceManager,
                                         RESOURCE_TYPES, NULL_RESOURCE_TYPE, ResourceError)


class Component:
    def __init__(self, name):
        self.name = name

    def producer(self):
        return 'resources!'


def test_resource_group():
    c = Component('base')
    r_type = 'column'
    r_names = [str(i) for i in range(5)]
    r_producer = c.producer
    r_dependencies = []

    rg = ResourceGroup(r_type, r_names, r_producer, r_dependencies)

    assert rg.type == r_type
    assert rg.names == [f'{r_type}.{name}' for name in r_names]
    assert rg.producer == c.producer
    assert not rg.dependencies
    assert list(rg) == rg.names


def test_resource_manager_get_resource_group():
    rm = ResourceManager()
    c = Component('base')
    r_type = 'column'
    r_names = [str(i) for i in range(5)]
    r_producer = c.producer
    r_dependencies = []

    rg = rm._get_resource_group(r_type, r_names, r_producer, r_dependencies)

    assert rg.type == r_type
    assert rg.names == [f'{r_type}.{name}' for name in r_names]
    assert rg.producer == c.producer
    assert not rg.dependencies
    assert list(rg) == rg.names


def test_resource_manager_get_resource_group_null():
    rm = ResourceManager()
    c = Component('base')
    r_type = 'column'
    r_names = []
    r_producer = c.producer
    r_dependencies = []

    rg = rm._get_resource_group(r_type, r_names, r_producer, r_dependencies)

    assert rg.type == NULL_RESOURCE_TYPE
    assert rg.names == [f'{NULL_RESOURCE_TYPE}.0']
    assert rg.producer == c.producer
    assert not rg.dependencies
    assert list(rg) == rg.names


def test_resource_manager_add_resources_bad_type():
    rm = ResourceManager()
    c = Component('base')
    r_type = 'unknown'
    r_names = [str(i) for i in range(5)]
    r_producer = c.producer
    r_dependencies = []

    with pytest.raises(ResourceError, match='Unknown resource type'):
        rm.add_resources(r_type, r_names, r_producer, r_dependencies)


def test_resource_manager_add_resources_multiple_producers():
    rm = ResourceManager()
    c1 = Component('1')
    c2 = Component('2')
    r_type = 'column'
    r1_names = [str(i) for i in range(5)]
    r2_names = [str(i) for i in range(5, 10)] + ['1']
    r1_producer = c1.producer
    r2_producer = c2.producer
    r_dependencies = []

    rm.add_resources(r_type, r1_names, r1_producer, r_dependencies)
    with pytest.raises(ResourceError, match='producers for column.1'):
        rm.add_resources(r_type, r2_names, r2_producer, r_dependencies)


def test_resource_manager_add_resources():
    rm = ResourceManager()
    for r_type in RESOURCE_TYPES:
        old_names = []
        for i in range(5):
            c = Component(f'r_type_{i}')
            names = [f'r_type_{i}_{j}' for j in range(5)]
            rm.add_resources(r_type, names, c.producer, old_names)
            old_names = names


def test_resource_manager_sorted_nodes_two_node_cycle():
    rm = ResourceManager()
    c = Component('test')

    rm.add_resources('column', ['1'], c.producer, ['stream.2'])
    rm.add_resources('stream', ['2'], c.producer, ['column.1'])

    with pytest.raises(ResourceError, match='cycle'):
        _ = rm.sorted_nodes


def test_resource_manager_sorted_nodes_three_node_cycle():
    rm = ResourceManager()
    c = Component('test')

    rm.add_resources('column', ['1'], c.producer, ['stream.3'])
    rm.add_resources('stream', ['2'], c.producer, ['column.1'])
    rm.add_resources('stream', ['3'], c.producer, ['stream.2'])

    with pytest.raises(ResourceError, match='cycle'):
        _ = rm.sorted_nodes


def test_resource_manager_sorted_nodes_large_cycle():
    rm = ResourceManager()
    c = Component('test')

    for i in range(10):
        rm.add_resources('column', [f'{i}'], c.producer, [f'column.{i%10}'])

    with pytest.raises(ResourceError, match='cycle'):
        _ = rm.sorted_nodes


def test_resource_manager_sorted_nodes_diamond():
    rm = ResourceManager()
    c = Component('test')

    rm.add_resources('column', ['1'], c.producer, [])
    rm.add_resources('column', ['2'], c.producer, ['column.1'])
    rm.add_resources('column', ['3'], c.producer, ['column.1'])
    rm.add_resources('column', ['4'], c.producer, ['column.2', 'column.3'])

    n = [str(node) for node in rm.sorted_nodes]

    assert n.index('(column.1)') < n.index('(column.2)')
    assert n.index('(column.1)') < n.index('(column.3)')
    assert n.index('(column.1)') < n.index('(column.4)')

    assert n.index('(column.2)') < n.index('(column.4)')
    assert n.index('(column.3)') < n.index('(column.4)')
