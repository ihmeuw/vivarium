import pandas as pd
import numpy as np

from vivarium.framework.event import Event, EventManager


def test_split_event():
    index1 = pd.Index(range(10))
    index2 = pd.Index(range(5))

    e1 = Event(index1)
    e2 = e1.split(index2)

    assert e1.index is index1
    assert e2.index is index2


def test_emission():
    signal = [False]

    def listener(*_, **__):
        signal[0] = True

    manager = EventManager()
    manager.clock = lambda: pd.Timestamp(1990, 1, 1)
    manager.step_size = lambda: pd.Timedelta(30, unit='D')
    emitter = manager.get_emitter('test_event')
    manager.register_listener('test_event', listener)
    emitter(Event(None))

    assert signal[0]

    signal[0] = False

    emitter = manager.get_emitter('test_unheard_event')
    emitter(Event(None))
    assert not signal[0]


def test_listener_priority():
    signal = [False, False, False]

    def listener1(*_, **__):
        signal[0] = True
        assert not signal[1]
        assert not signal[2]

    def listener2(*_, **__):
        signal[1] = True
        assert signal[0]
        assert not signal[2]

    def listener3(*_, **__):
        signal[2] = True
        assert signal[0]
        assert signal[1]

    manager = EventManager()
    manager.clock = lambda: pd.Timestamp(1990, 1, 1)
    manager.step_size = lambda: pd.Timedelta(30, 'D')
    emitter = manager.get_emitter('test_event')
    manager.register_listener('test_event', listener1, priority=0)
    manager.register_listener('test_event', listener2)
    manager.register_listener('test_event', listener3, priority=9)

    emitter(Event(None))
    assert np.all(signal)


def test_contains():
    event = 'test_event'

    manager = EventManager()
    assert event not in manager
    manager.get_emitter(event)
    assert event in manager
