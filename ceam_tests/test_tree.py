import pytest

from ceam.tree import Node

def test_tree_construction():
    a = Node()
    b = Node()
    c = Node()

    assert len(a.children) == 0
    assert a.parent is None

    a.add_child(b)

    assert len(a.children) == 1
    assert b in a.children
    assert a.parent is None
    assert b.parent is a
    assert b.root is a

    c.add_child(a)

    assert a.parent is c
    assert len(a.children) == 1
    assert len(c.children) == 1
    assert b.root == a.root == c

def test_tree_deconstruction():
    a = Node()
    b = Node()
    c = Node()
    d = Node()

    a.add_children([b,c])
    b.add_child(d)

    assert b in a.children
    assert b.parent is a
    assert b.root is a
    assert d.root is a

    a.remove_child(b)

    assert b not in a.children
    assert b.parent is None
    assert b.root is b
    assert c in a.children
    assert c.parent is a
    assert d.root is b

class FancyNode(Node):
    pass

def test_decendent_filtering():
    a = Node()
    b = Node()
    b.test_attribute = 'thing'
    c = FancyNode()
    c.test_attribute = 'thing'
    d = FancyNode()

    a.add_child(c)
    c.add_children([b,d])

    assert set(a.all_decendents()) == {b, c, d}
    assert set(a.all_decendents(of_type=FancyNode)) == {c, d}
    assert set(a.all_decendents(with_attr='test_attribute')) == {b, c}

def test_child_filtering():
    a = Node()
    b = Node()
    b.test_attribute = 'thing'
    c = FancyNode()
    c.test_attribute = 'thing'
    d = FancyNode()

    a.add_child(c)
    c.add_children([b,d])

    assert set(a.all_children()) == {c}
