import pytest

from vivarium.framework.lifecycle import (LifeCycleState, LifeCyclePhase, LifeCycle,
                                          LifeCycleError, ConstraintError,
                                          LifeCycleManager, ConstraintMaker)


def test_state_add_next():
    a = LifeCycleState('a')
    b = LifeCycleState('b')
    c = LifeCycleState('c')

    assert a.valid_next_state(None)
    a.add_next(b)
    assert a.valid_next_state(b)
    assert not a.valid_next_state(c)
    assert not a.valid_next_state(None)

    assert b.valid_next_state(None)
    b.add_next(a, loop=True)
    assert b.valid_next_state(a)
    assert b.valid_next_state(None)
    b.add_next(c)
    assert not b.valid_next_state(None)
    assert b.valid_next_state(a)
    assert b.valid_next_state(c)

    assert c.valid_next_state(None)
    assert not c.valid_next_state(a)
    assert not c.valid_next_state(b)


def test_state_entrance_count():
    a = LifeCycleState('a')
    assert a.entrance_count == 0
    a.enter()
    a.enter()
    assert a.entrance_count == 2


class BumbleBee:

    def __init__(self, name):
        self.name = name

    def buzz(self):
        return "Buzzzzz"

    def what_we_doin(self):
        return "Gettin' honey."

    def __call__(self):
        return  "On the way!"


def test_state_add_handlers():
    a = LifeCycleState('a')
    b = BumbleBee('Alice')
    a.add_handlers([b.buzz, b.what_we_doin, test_state_add_handlers])
    assert a._handlers == ['BumbleBee(Alice).buzz', 'BumbleBee(Alice).what_we_doin',
                           'Unbound function test_state_add_handlers']


def test_phase_init_no_loop():
    phase_name = 'test'
    state_names = ['a', 'b', 'c', 'd', 'e']

    p = LifeCyclePhase(phase_name, state_names, loop=False)
    assert p.name == phase_name
    assert [s.name for s in p.states] == state_names
    current = p.states[0]
    for state in p.states[1:]:
        assert current.valid_next_state(state)
        current = state
    assert current.valid_next_state(None)


def test_phase_init_loop():
    phase_name = 'test'
    state_names = ['a', 'b', 'c', 'd', 'e']

    p = LifeCyclePhase(phase_name, state_names, loop=True)
    assert p.name == phase_name
    assert [s.name for s in p.states] == state_names
    current = p.states[0]
    for state in p.states[1:]:
        assert current.valid_next_state(state)
        current = state
    assert current.valid_next_state(p.states[0])


def test_phase_add_next():
    phase1_name = 'test1'
    phase2_name = 'test2'
    states1 = ['a', 'b', 'c', 'd', 'e']
    states2 = ['1', '2', '3', '4', '5']

    p1 = LifeCyclePhase(phase1_name, states1, loop=True)
    p2 = LifeCyclePhase(phase2_name, states2, loop=False)

    p1.add_next(p2)
    assert p1.states[-1].valid_next_state(p1.states[0])
    assert p1.states[-1].valid_next_state(p2.states[0])


def test_phase_get_state():
    name = 'test'
    state_names = ['a', 'b', 'c', 'd', 'e']

    p = LifeCyclePhase(name, state_names, loop=False)

    for name, state in zip(state_names, p.states):
        assert state is p.get_state(name)

    with pytest.raises(IndexError):
        p.get_state('not_a_state')


def test_phase_contains():
    name = 'test'
    state_names = ['a', 'b', 'c', 'd', 'e']

    p = LifeCyclePhase(name, state_names, loop=False)

    for name in state_names:
        assert name in p

    for name in ['1', '2', '3', '']:
        assert name not in p


def test_lifecycle_init():
    lc = LifeCycle()
    assert 'initialization' in lc
    assert 'not_a_state' not in lc


def test_lifecycle_validate():
    lc = LifeCycle()

    with pytest.raises(LifeCycleError, match='phase names must be unique'):
        lc._validate('initialization', ['a', 'b', 'c'])

    with pytest.raises(LifeCycleError, match='duplicate state names'):
        lc._validate('new_phase', ['a', 'b', 'c', 'b'])

    with pytest.raises(LifeCycleError, match='initialization'):
        lc._validate('new_phase', ['a', 'b', 'initialization'])

    lc._validate('new_phase', ['a', 'b', 'c'])


@pytest.mark.parametrize('loop', [True, False])
def test_lifecycle_add_phase_fail(loop):
    lc = LifeCycle()

    with pytest.raises(LifeCycleError, match='phase names must be unique'):
        lc.add_phase('initialization', ['a', 'b', 'c'], loop)

    with pytest.raises(LifeCycleError, match='duplicate state names'):
        lc.add_phase('new_phase', ['a', 'b', 'c', 'b'], loop)

    with pytest.raises(LifeCycleError, match='initialization'):
        lc.add_phase('new_phase', ['a', 'b', 'initialization'], loop)


@pytest.mark.parametrize('loop', [True, False])
def test_lifecycle_add_phase(loop):
    lc = LifeCycle()
    lc.add_phase('phase1', ['a', 'b', 'c'], loop)


def test_lifecycle_get_state():
    lc = LifeCycle()
    assert lc.get_state('initialization').name == 'initialization'

    with pytest.raises(LifeCycleError, match='non-existent state'):
        lc.get_state('not_a_state')


def test_lifecycle_get_states():
    lc = LifeCycle()
    assert lc.get_state_names('initialization') == ['initialization']

    with pytest.raises(LifeCycleError, match='non-existent phase'):
        lc.get_state_names('not_a_phase')


def test_lifecycle_integration():
    lc = LifeCycle()

    with pytest.raises(LifeCycleError, match='non-existent state'):
        lc.get_state('a')
    with pytest.raises(LifeCycleError, match='non-existent phase'):
        lc.get_state_names('phase1')

    init = lc.get_state('initialization')
    assert init.valid_next_state(None)

    lc.add_phase('phase1', ['a'], loop=False)
    a = lc.get_state('a')

    assert lc.get_state_names('phase1') == ['a']
    assert init.valid_next_state(a)
    assert not init.valid_next_state(None)
    assert a.valid_next_state(None)

    lc.add_phase('phase2', ['b'], loop=True)
    b = lc.get_state('b')

    assert lc.get_state_names('phase2') == ['b']
    assert a.valid_next_state(b)
    assert b.valid_next_state(None)
    assert b.valid_next_state(b)

    lc.add_phase('phase3', ['c', 'd'], loop=True)
    c = lc.get_state('c')
    d = lc.get_state('d')

    assert not b.valid_next_state(None)
    assert b.valid_next_state(b)
    assert b.valid_next_state(c)
    assert c.valid_next_state(d)
    assert d.valid_next_state(c)
    assert d.valid_next_state(None)

    with pytest.raises(LifeCycleError, match='phase names must be unique'):
        lc.add_phase('phase2', ['1', '2'], loop=True)

    with pytest.raises(LifeCycleError, match='duplicate state names'):
        lc.add_phase('phase4', ['1', '2', '3', '2'], loop=False)

    with pytest.raises(LifeCycleError, match='state names must be unique'):
        lc.add_phase('phase4', ['a', 'f', 'g'], loop=False)

    with pytest.raises(LifeCycleError, match='non-existent state'):
        lc.get_state('not a state')

    with pytest.raises(LifeCycleError, match='non-existent phase'):
        lc.get_state_names('not a phase')


def test_manager_init():
    lm = LifeCycleManager()
    assert lm.name == 'life_cycle_manager'
    assert lm.current_state == 'initialization'


@pytest.mark.parametrize('loop', [True, False])
def test_lifecycle_manager_add_phase_fail(loop):
    lm = LifeCycleManager()

    with pytest.raises(LifeCycleError, match='phase names must be unique'):
        lm.add_phase('initialization', ['a', 'b', 'c'], loop)

    with pytest.raises(LifeCycleError, match='duplicate state names'):
        lm.add_phase('new_phase', ['a', 'b', 'c', 'b'], loop)

    with pytest.raises(LifeCycleError, match='initialization'):
        lm.add_phase('new_phase', ['a', 'b', 'initialization'], loop)


@pytest.mark.parametrize('loop', [True, False])
def test_lifecycle_manager_add_phase(loop):
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a', 'b', 'c'], loop)


def test_lifecycle_manager_get_states():
    lm = LifeCycleManager()
    assert lm.get_state_names('initialization') == ['initialization']

    with pytest.raises(LifeCycleError, match='non-existent phase'):
        lm.get_state_names('not_a_phase')


def test_lifecycle_manager_set_state():
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a', 'b'])
    lm.add_phase('phase2', ['c', 'd'])

    assert lm.current_state == 'initialization'
    lm.set_state('a')
    assert lm.current_state == 'a'
    lm.set_state('b')
    assert lm.current_state == 'b'

    with pytest.raises(LifeCycleError, match='Invalid transition'):
        lm.set_state('a')  # phase1 does not loop

    with pytest.raises(LifeCycleError, match='Invalid transition'):
        lm.set_state('d')  # next state is c

    with pytest.raises(LifeCycleError, match='non-existent state'):
        lm.set_state('e')

    lm.set_state('c')
    assert lm.current_state == 'c'
    lm.set_state('d')
    assert lm.current_state == 'd'


def test_lifecycle_manager_set_state_with_loop():
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a', 'b'], loop=True)
    lm.add_phase('phase2', ['c', 'd'])

    assert lm.current_state == 'initialization'
    lm.set_state('a')
    assert lm.current_state == 'a'
    lm.set_state('b')
    assert lm.current_state == 'b'

    lm.set_state('a')  # phase permits loops
    assert lm.current_state == 'a'
    lm.set_state('b')
    assert lm.current_state == 'b'

    with pytest.raises(LifeCycleError, match='Invalid transition'):
        lm.set_state('d')  # next state is a or c

    with pytest.raises(LifeCycleError, match='non-existent state'):
        lm.set_state('e')

    lm.set_state('c')
    assert lm.current_state == 'c'
    lm.set_state('d')
    assert lm.current_state == 'd'

    with pytest.raises(LifeCycleError, match='Invalid transition'):
        lm.set_state('c')  # phase 2 does not permit loops


def test_lifecycle_manager_add_handlers():
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a'])

    init = lm.lifecycle.get_state('initialization')
    a = lm.lifecycle.get_state('a')

    b = BumbleBee('Alice')

    lm.add_handlers('initialization', [b.buzz, b.what_we_doin, test_lifecycle_manager_add_handlers])
    assert init._handlers == ['BumbleBee(Alice).buzz',
                              'BumbleBee(Alice).what_we_doin',
                              'Unbound function test_lifecycle_manager_add_handlers']

    lm.add_handlers('a', [b.buzz])
    assert a._handlers == ['BumbleBee(Alice).buzz']


def test_lifecycle_manager_add_constraint_fail():
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a', 'b', 'c'])

    alice = BumbleBee('Alice')

    with pytest.raises(ValueError, match='Must provide exactly one'):
        lm.add_constraint(alice.buzz)

    with pytest.raises(ValueError, match='Must provide exactly one'):
        lm.add_constraint(alice.buzz, allow_during=['initialization', 'a'], restrict_during=['c'])

    with pytest.raises(LifeCycleError, match='states not in the life cycle'):
        lm.add_constraint(alice.buzz, allow_during=['not_a_state'])

    with pytest.raises(LifeCycleError, match='states not in the life cycle'):
        lm.add_constraint(alice.buzz, restrict_during=['not_a_state'])

    lm.add_constraint(alice.buzz, allow_during=['a'])

    with pytest.raises(ConstraintError, match='already been constrained'):
        lm.add_constraint(alice.buzz, allow_during=['b'])

    with pytest.raises(TypeError, match='bound object methods'):
        lm.add_constraint(test_lifecycle_manager_add_constraint_fail, allow_during=['a'])

    with pytest.raises(ValueError, match='normal object methods'):
        lm.add_constraint(alice.__call__, allow_during=['a'])


def test_lifecycle_manager_add_constraint():
    lm = LifeCycleManager()
    lm.add_phase('phase1', ['a', 'b', 'c'], loop=True)

    alice = BumbleBee('Alice')
    bob = BumbleBee('Bob')

    lm.add_constraint(alice.buzz, allow_during=['initialization', 'a'])
    lm.add_constraint(bob.buzz, restrict_during=['initialization', 'c'])

    alice.buzz()
    with pytest.raises(ConstraintError, match='it may only be called during'):
        bob.buzz()

    lm.set_state('a')
    alice.buzz()
    bob.buzz()

    lm.set_state('b')
    with pytest.raises(ConstraintError, match='it may only be called during'):
        alice.buzz()
    bob.buzz()

    lm.set_state('c')
    with pytest.raises(ConstraintError, match='it may only be called during'):
        alice.buzz()
    with pytest.raises(ConstraintError, match='it may only be called during'):
        bob.buzz()

    lm.set_state('a')
    alice.buzz()
    bob.buzz()
