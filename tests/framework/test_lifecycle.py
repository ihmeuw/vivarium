from vivarium.framework.lifecycle import (LifeCycleState, LifeCyclePhase, LifeCycle,
                                          LifeCycleError, ConstraintError,
                                          LifeCycleManager, ConstraintMaker)


def test_state_add_next():
    a = LifeCycleState('a')
    b = LifeCycleState('b')
    c = LifeCycleState('c')
    a.add_next(b)
    b.add_next(a, loop=True)
    b.add_next(c)
    assert a.valid_next_state(b)
    assert not a.valid_next_state(c)
    assert b.valid_next_state(a)
    assert b.valid_next_state(c)
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


def test_state_add_handlers():
    a = LifeCycleState('a')
    b = BumbleBee('Arthur')
    a.add_handlers([b.buzz, b.what_we_doin, test_state_add_handlers])
    assert a._handlers == ['BumbleBee(Arthur).buzz', 'BumbleBee(Arthur).what_we_doin',
                           'Unbound function test_state_add_handlers']






