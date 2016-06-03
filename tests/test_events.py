# ~/ceam/tests/test_events.py

from unittest import TestCase

import numpy as np
import pandas as pd

from ceam.events import EventHandler, Event, PopulationEvent, only_living

class TestEventHandler(TestCase):
    def test_listener_registration(self):
        eh = EventHandler()
        trigger = [False]
        def listener(event):
            trigger[0] = True
        eh.register_event_listener(listener, 'test')
        eh.emit_event(Event('test'))
        self.assertTrue(trigger[0])

        trigger[0] = False
        eh.deregister_event_listener(listener, 'test')
        eh.emit_event(Event('test'))
        self.assertFalse(trigger[0])

    def test_generic_listener(self):
        eh = EventHandler()
        records = []
        eh.register_event_listener(lambda e: records.append('label'), 'test')
        eh.register_event_listener(lambda e: records.append('generic'))
        eh.emit_event(Event('test'))
        self.assertListEqual(records, ['label', 'generic'])

        records = []
        eh.emit_event(Event('other_test'))
        self.assertListEqual(records, ['generic'])

    def test_listener_priority(self):
        eh = EventHandler()
        records = []
        eh.register_event_listener(lambda e: records.append('second'), 'test', priority=1)
        eh.register_event_listener(lambda e: records.append('first'), 'test', priority=0)
        eh.register_event_listener(lambda e: records.append('last'), 'test', priority=9)
        eh.emit_event(Event('test'))
        self.assertListEqual(records, ['first', 'second', 'last'])


class TestEventListenerDecorators(TestCase):
    def test_only_living(self):
        pop = pd.DataFrame({'alive': [True, True, False, True]})

        @only_living
        def inner(event):
            self.assertEqual(len(event.affected_population), 3)
        inner(PopulationEvent('test', pop))

# End.
