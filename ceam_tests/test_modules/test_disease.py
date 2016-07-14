import pytest
from unittest.mock import Mock

import pandas as pd
import numpy as np

from ceam.state_machine import Transition, State

from ceam.modules.disease import DiseaseState

def test_dwell_time():
    test_state = DiseaseState('test', 0.0, dwell_time=10)
    test_state.parent = Mock()
    test_state.parent.parent = None
    test_state.parent.current_time.timestamp.return_value = 0

    done_state = State('done')
    test_state.transition_set.add(Transition(done_state))

    population = pd.DataFrame({'state': ['test']*10, 'test_event_time': [0]*10, 'test_event_count': [0]*10})

    population = test_state.next_state(population, 'state')
    assert np.all(population.state == 'test')

    test_state.root.current_time.timestamp.return_value = 20

    population = test_state.next_state(population, 'state')
    assert np.all(population.state == 'done')
    assert np.all(population.test_event_time == 20)
    assert np.all(population.test_event_count == 1)
