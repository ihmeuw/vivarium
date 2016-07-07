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
