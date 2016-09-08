from ceam.framework.engine import run_configuration, configure

def workflow(component_config, sub_configuration_name, results_path, draw_number, config):
    configure(draw_number=draw_number, simulation_config=config)
    run_configuration(component_config, results_path)
