import os
from vivarium import config

# Remove user overrides but keep custom cache locations if any
config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path', 'input_data.auxiliary_data_folder'])
config.simulation_parameters.set_with_metadata('year_start', 1990, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('year_end', 2010, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('time_step', 30.5, layer='override', source=os.path.realpath(__file__))
config.simulation_parameters.set_with_metadata('initial_age', '', layer='override', source=os.path.realpath(__file__))
