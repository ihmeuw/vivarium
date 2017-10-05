import os
from time import time
import logging

import pandas as pd
import numpy as np

from celery import Celery
from billiard import current_process


app = Celery()


@app.task(autoretry_for=(Exception,), max_retries=2)
def worker(parameters, logging_directory):
    input_draw = parameters['input_draw']
    model_draw = parameters['model_draw']
    component_config = parameters['components']
    branch_config = parameters['config']

    np.random.seed([input_draw, model_draw])
    worker_ = current_process().index
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        filename=os.path.join(logging_directory, str(worker_)+'.log'), level=logging.DEBUG)
    logging.info('Starting job: {}'.format((input_draw, model_draw, component_config, branch_config)))

    try:
        from vivarium.framework.engine import build_simulation_configuration, run, setup_simulation
        from vivarium.framework.components import load_component_manager
        from vivarium.framework.util import collapse_nested_dict

        config = build_simulation_configuration(parameters)
        config.run_configuration.run_id = str(worker_)+'_'+str(time())
        if branch_config is not None:
            config.run_configuration.update({'run_key': dict(branch_config)}, layer='override', source=str(worker_))
            config.run_configuration.run_key.update({'input_draw': input_draw,
                                                     'model_draw': model_draw})

        component_manager = load_component_manager(config)
        simulation = setup_simulation(component_manager, config)
        results = run(simulation)
        idx = pd.MultiIndex.from_tuples([(input_draw, model_draw)],
                                        names=['input_draw_number', 'model_draw_number'])
        results = pd.DataFrame(results, index=idx).to_json()

        return results
    except Exception:
        logging.exception('Unhandled exception in worker')
        raise
    finally:
        logging.info('Exiting job: {}'.format((input_draw, model_draw, component_config, branch_config)))

