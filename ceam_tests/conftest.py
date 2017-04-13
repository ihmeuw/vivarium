from ceam import config

# Remove user overrides but keep custom cache locations if any
config.reset_layer('override', preserve_keys=['input_data.intermediary_data_cache_path', 'input_data.auxiliary_data_folder'])
config.simulation_parameters.year_start = 1990
config.simulation_parameters.year_end = 2010
config.simulation_parameters.time_step = 30.5
