import os

import pytest

from vivarium.framework.configuration import build_simulation_configuration


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration()
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    config.update({
        'time': {
            'start': {
                'year': 1990,
            },
            'end': {
                'year': 2010
            },
            'step_size': 30.5
        }
    }, **metadata)
    return config
