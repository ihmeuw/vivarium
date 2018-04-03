import os

import pytest

from vivarium.framework.engine import build_simulation_configuration


@pytest.fixture(scope='module')
def base_config():
    config = build_simulation_configuration({})
    metadata = {'layer': 'override', 'source': os.path.realpath(__file__)}
    config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path',
                                                   'input_data.auxiliary_data_folder'])
    config.time.start.set_with_metadata('year', 1990, **metadata)
    config.time.end.set_with_metadata('year', 2010, **metadata)
    config.time.set_with_metadata('step_size', 30.5, **metadata)
    return config
