from bdb import BdbQuit
import os

import click

from vivarium.framework.engine import run_simulation

from .utilities import verify_yaml

import logging
logging.getLogger(__name__)


@click.group()
def simulate():
    pass


@simulate.command()
@click.argument('model_specification', callback=verify_yaml,
                type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option('--results_directory', '-o', type=click.Path(resolve_path=True),
              default=os.path.expanduser('~/vivarium_results/'),
              help='Top level output directory to write results to')
@click.option('--verbose', '-v', is_flag=True, help='Report each time step')
@click.option('--log', type=click.Path(dir_okay=False, resolve_path=True), help='Path to log file')
@click.option('--pdb', 'with_debugger', is_flag=True, help='Drop into python debugger if an error occurs.')
def run(model_specification, results_directory, verbose, log, with_debugger):
    log_level = logging.DEBUG if verbose else logging.ERROR
    logging.basicConfig(filename=log, level=log_level)

    try:
        run_simulation(model_specification, results_directory)
    except (BdbQuit, KeyboardInterrupt):
        raise
    except Exception as e:
        if with_debugger:
            import pdb
            import traceback
            traceback.print_exc()
            pdb.post_mortem()
        else:
            logging.exception("Uncaught exception {}".format(e))
            raise
