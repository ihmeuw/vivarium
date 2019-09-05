"""
=====================
Dependency Management
=====================

This module provides a tool to manage dependencies on resources within a
``Vivarium`` simulation. These resources take the form of things that can be
created and utilized by components, for example columns in the
:mod:`state table <vivarium.framework.population>`
or :mod:`named value pipelines <vivarium.framework.values>`. Because these
need to be created before they can be used, they are sensitive to ordering.
The intent behind this tool is to provide an interface that allows other
managers to register resources with the dependency manager and in turn ask for
ordered sequences of these resources according to their dependencies or raise
exceptions if this is not possible.

"""
from typing import Sequence, List, Tuple, Callable, Union
from collections import deque

import networkx as nx

from vivarium.exceptions import VivariumError


class DependencyError(VivariumError):
    """Error raised when a dependency requirement is violated"""
    pass


class ResourceProducer:

    def __init__(self, resource_type, resource_names, producer, dependencies):
        self.resource_type = resource_type
        self.resource_names = resource_names
        self.producer = producer
        self.dependencies = dependencies


class EmptySet:

    def add(self, item):
        pass

    def __contains__(self, item):
        return False


class ResourceGroup:

    def __init__(self, resource_type, single_producer=False):
        self.type = resource_type
        # One initializer per component, maybe multiple value producers
        self.producer_components = set() if single_producer else EmptySet()
        self.resources = {}

    def add_resources(self, resource_type, resource_names, producer, dependencies):
        if producer.__self__.name in self.producer_components:
            raise  # Component has more than one producer for resource type ...
        self.producer_components.add(producer.__self__.name)

        producer = ResourceProducer(resource_type, resource_names, producer, dependencies)
        for resource_name in resource_names:
            key = f'{resource_type}.{resource_name}'
            if key in self.resources:
                raise  # More than one producer for resource ...
            self.resources[key] = producer

    def __iter__(self):
        for resource_producer in self._to_graph():
            yield resource_producer.producer

    def _to_graph(self):
        g = nx.DiGraph()
        g.add_nodes_from(set([r for r in self.resources.values() if r.type == self.type]))

        def _add_in_edges_to(node, from_keys):
            for dependency_key in from_keys:
                d = self.resources[dependency_key]
                if d.type == self.type:
                    try:
                        g.add_edge(d, node)
                    except:  # Some kind of node doesn't exist error
                        raise  # Resource r depends on d but d has no producer.
                else:
                    _add_in_edges_to(node, from_keys=d.dependencies)

        for r in set(self.resources.values()):
            _add_in_edges_to(r, r.dependencies)

        return nx.algorithms.topological_sort(g)


class ResourceManager:

    def __init__(self):
        self._resource_groups = {'column': ResourceGroup('column', single_producer=True)}

    @property
    def name(self):
        return "resource_manager"

    def register_resource_producer(self, resource_names, resource_type, producer, dependencies):
        self._resource_groups[resource_type].add_resources(resource_names, resource_type, producer, dependencies)

    def get_resource_producers(self, resource_type):
        return lambda: self._resource_groups[resource_type]


class ResourceInterface:

    def __init__(self, manager: ResourceManager):
        self._manager = manager

    def register_resource_producer(self, resource_names, resource_type, producer, dependencies):
        self._manager.register_resource_producer(resource_names, resource_type, producer, dependencies)

    def get_resource_producers(self, resource_type):
        return self._manager.get_resource_producers(resource_type)
