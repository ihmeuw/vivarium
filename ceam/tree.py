# ~/ceam/ceam/tree.py


class Node:
    @property
    def children(self):
        if not hasattr(self, '_children'):
            self._children = set()
        return self._children

    @property
    def parent(self):
        if not hasattr(self, '_parent'):
            self._parent = None
        return self._parent

    @parent.setter
    def parent(self, value):
        self._parent = value

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

    def _filter_decendents(self, of_type, with_attr, max_depth, current_depth):
        results = []
        for child in self.children:
            if of_type is None or isinstance(child, of_type):
                if with_attr is None or hasattr(child, with_attr):
                    results.append(child)
            if max_depth > current_depth:
                results.extend(child._filter_decendents(of_type, with_attr, max_depth=max_depth, current_depth=current_depth+1))
        return results

    def all_children(self, of_type=None, with_attr=None):
        return self._filter_decendents(of_type, with_attr, max_depth=0, current_depth=0)

    def all_decendents(self, of_type=None, with_attr=None):
        return self._filter_decendents(of_type, with_attr, max_depth=float('inf'), current_depth=0)

    @property
    def root(self):
        if self.parent is None:
            return self
        else:
            root = self.parent
            while root.parent is not None:
                root = root.parent
            return root


# End.
