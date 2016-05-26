from unittest import TestCase

from ceam.engine import EventHandler

class TestEventHandler(TestCase):
    def test_listener_registration(self):
        eh = EventHandler()
        trigger = [False]
        def listener(label, mask, simulation):
            trigger[0] = True
            self.assertListEqual(mask, [True, False])
            self.assertEqual(simulation, 'mock_simulation')
        eh.register_event_listener(listener, 'test')
        eh.emit_event('test', [True, False], 'mock_simulation')
        self.assertTrue(trigger[0])

        trigger[0] = False
        eh.deregister_event_listener(listener, 'test')
        eh.emit_event('test', [], None)
        self.assertFalse(trigger[0])
    
    def test_generic_listener(self):
        eh = EventHandler()
        records = []
        eh.register_event_listener(lambda a,b,c: records.append('label'), 'test')
        eh.register_event_listener(lambda a,b,c: records.append('generic'))
        eh.emit_event('test', [], None)
        self.assertListEqual(records, ['label', 'generic'])

        records = []
        eh.emit_event('other_test', [], None)
        self.assertListEqual(records, ['generic'])

    def test_listener_priority(self):
        eh = EventHandler()
        records = []
        eh.register_event_listener(lambda a,b,c: records.append('second'), 'test', priority=1)
        eh.register_event_listener(lambda a,b,c: records.append('first'), 'test', priority=0)
        eh.register_event_listener(lambda a,b,c: records.append('last'), 'test', priority=9)
        eh.emit_event('test', [], None)
        self.assertListEqual(records, ['first', 'second', 'last'])
