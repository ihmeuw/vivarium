class NodeBehaviorMixin:
    pass

class Node:
    def __init__(self):
        self.children = set()
        self.parent = None

    def add_child(self, node):
        self.children.add(node)
        node.added_to(self)

    def add_children(self, nodes):
        for node in nodes:
            self.add_child(node)

    def added_to(self, node):
        self.parent = node

    def remove_child(self, node):
        self.children.remove(node)
        node.removed_from(self)

    def remove_children(self, nodes):
        for node in nodes:
            self.remove_child(node)

    def removed_from(self, node):
        self.parent = None

    def all_decendents(self, of_type=object):
        results = []
        for child in self.children:
            if isinstance(child, of_type):
                results.append(child)
            results.extend(child.all_decendents(of_type))
        return results

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            root = self.parent
            while root.parent is not None:
                root = root.parent
            return root


class Root(Node):
    pass
