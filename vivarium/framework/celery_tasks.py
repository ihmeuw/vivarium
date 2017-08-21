import os
from time import time
import logging

import pandas as pd
import numpy as np

from celery import Celery
from billiard import current_process


app = Celery()

@app.task(autoretry_for=(Exception,), max_retries=2)
def worker(input_draw_number, model_draw_number, component_config, branch_config, logging_directory):
    np.random.seed([input_draw_number, model_draw_number])
    worker = current_process().index
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', filename=os.path.join(logging_directory, str(worker)+'.log'), level=logging.DEBUG)
    logging.info('Starting job: {}'.format((input_draw_number, model_draw_number, component_config, branch_config)))

    run_configuration = component_config['configuration'].get('run_configuration', {})
    results_directory = run_configuration['results_directory']
    run_configuration['run_id'] = str(worker)+'_'+str(time())
    if branch_config is not None:
        run_configuration['run_key'] = dict(branch_config)
        run_configuration['run_key']['input_draw'] = input_draw_number
        run_configuration['run_key']['model_draw'] = model_draw_number
    component_config['configuration']['run_configuration'] = run_configuration

    try:
        from vivarium.framework.engine import configure, run
        from vivarium.framework.components import load_component_manager
        from vivarium.framework.util import collapse_nested_dict

        configure(input_draw_number=input_draw_number, model_draw_number=model_draw_number, simulation_config=branch_config)
        component_maneger = load_component_manager(component_config)
        results = run(component_manager)
        idx=pd.MultiIndex.from_tuples([(input_draw_number, model_draw_number)], names=['input_draw_number','model_draw_number'])
        results = pd.DataFrame(results, index=idx).to_json()

        return results
    except Exception as e:
        logging.exception('Unhandled exception in worker')
        raise
    finally:
        logging.info('Exiting job: {}'.format((input_draw_number, model_draw_number, component_config, branch_config)))

