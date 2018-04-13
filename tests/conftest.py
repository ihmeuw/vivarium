import os

import pytest

from vivarium.framework.engine import build_simulation_configuration


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration().configuration
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                  'input_data.auxiliary_data_folder'])
    config.update(
        {
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
